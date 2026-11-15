$ErrorActionPreference = 'Stop'
$interop = @'
using System.Runtime.InteropServices;
public static class RynExecutionState {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
Add-Type -TypeDefinition $interop

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dataDir = (Resolve-Path (Join-Path $repo '..\ryn-data')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runDir = Join-Path $dataDir 'runs\ryn-0.3b-base-final-256'
$logPath = Join-Path $dataDir 'pretrain-continuation-256.log'
$batchSize, $gradAccum, $oomCount, $cublasFailures = 8, 4, 0, 0
$stayAwake = [Convert]::ToUInt32('80000001', 16)
$continuous = [Convert]::ToUInt32('80000000', 16)

if ([RynExecutionState]::SetThreadExecutionState($stayAwake) -eq 0) {
    throw 'Could not request that Windows stay awake during local training.'
}

try {
    Set-Location $repo
    while ($true) {
        $arguments = @(
            '-u', 'scripts/train.py',
            '--stage', 'pretrain', '--config', 'configs/ryn0_3b.json',
            '--data', '../ryn-data/pretrain.jsonl', '../ryn-data/pretrain.extra-train.jsonl',
            '--validation-data', '../ryn-data/pretrain.final-validation.jsonl',
            '--output-dir', '../ryn-data/runs/ryn-0.3b-base-final-256',
            '--target-tokens', '898419200', '--prior-tokens', '101580800', '--max-seq-len', '256',
            '--batch-size', "$batchSize", '--grad-accum', "$gradAccum",
            '--learning-rate', '4.94e-5', '--warmup-steps', '0',
            '--save-every', '15000', '--eval-every', '5000', '--eval-steps', '32', '--log-every', '100'
        )
        if (Test-Path (Join-Path $runDir 'latest.json')) {
            $arguments += '--resume'
        } else {
            $arguments += @('--init-from', '../ryn-data/runs/ryn-0.3b-base-final-512/checkpoints/checkpoint-00008900')
        }
        & $python @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) { break }
        if ($exitCode -eq 1) {
            $logTail = (Get-Content -LiteralPath $logPath -Tail 30) -join "`n"
            if ($logTail -match 'CUBLAS_STATUS_EXECUTION_FAILED') {
                if ($batchSize -eq 8) { $batchSize, $gradAccum = 4, 8 }
                elseif ($batchSize -eq 4) { $batchSize, $gradAccum = 2, 16 }
                elseif ($batchSize -eq 2) { $batchSize, $gradAccum = 1, 32 }
                elseif ($cublasFailures -ge 1) { throw "Repeated cuBLAS launch failure; latest checkpoint is preserved. See $logPath" }
                $oomCount = 0
                $cublasFailures++
                Write-Host 'cuBLAS launch failure; resuming from the latest checkpoint with the lower-memory batch grouping.'
                Start-Sleep -Seconds 5
                continue
            }
            throw "Training exited with code $exitCode; see $logPath"
        }
        if ($exitCode -ne 2) { throw "Training exited with code $exitCode; see $logPath" }

        $oomCount++
        if ($batchSize -eq 8 -and $oomCount -ge 1) {
            $batchSize, $gradAccum, $oomCount = 4, 8, 0
            Write-Host 'OOM at batch size 8; resuming with batch size 4 and accumulation 8.'
        } elseif ($batchSize -eq 4 -and $oomCount -ge 1) {
            $batchSize, $gradAccum, $oomCount = 2, 16, 0
            Write-Host 'OOM at batch size 4; resuming with batch size 2 and accumulation 16.'
        } elseif ($batchSize -eq 2 -and $oomCount -ge 2) {
            $batchSize, $gradAccum, $oomCount = 1, 32, 0
            Write-Host 'Repeated OOM; resuming with batch size 1 and accumulation 32.'
        } elseif ($batchSize -eq 1 -and $oomCount -ge 5) {
            throw "Five OOM retries at batch size 1; latest checkpoint is preserved. See $logPath"
        }
        Start-Sleep -Seconds 5
    }
} finally {
    [void][RynExecutionState]::SetThreadExecutionState($continuous)
}
