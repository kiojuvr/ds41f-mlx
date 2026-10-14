param([string]$BaseUrl = 'http://192.168.68.56:8000', [string]$OutputDirectory = 'artifacts/remote-lan-windows-20261014')
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$run = 'windows-lan-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$identity = [ordered]@{run_id=$run; started_utc=[DateTime]::UtcNow.ToString('o'); hostname=[Environment]::MachineName; os=[Environment]::OSVersion.VersionString; base_url=$BaseUrl; proxy_bypass='curl --noproxy *'; addresses=@(Get-NetIPAddress -AddressFamily IPv4 | Select-Object InterfaceAlias,IPAddress); route=(Find-NetRoute -RemoteIPAddress ([uri]$BaseUrl).Host | Out-String); checkout=(git rev-parse HEAD)}
$identity | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 "$OutputDirectory/client.json"
function Invoke-Probe($Name, $Path, $Body) {
    $curlArgs = @('--noproxy','*','--connect-timeout','5','--max-time','180','--silent','--show-error','--no-buffer','--dump-header',"$OutputDirectory/$Name.headers",'--output',"$OutputDirectory/$Name.body",'--write-out','%{json}')
    if ($Body) {
        $Body | ConvertTo-Json -Depth 8 -Compress | Set-Content -Encoding Ascii "$OutputDirectory/$Name.request.json"
        $curlArgs += @('-H','Content-Type: application/json','--data-binary',"@$OutputDirectory/$Name.request.json")
    }
    $start = [DateTime]::UtcNow.ToString('o')
    $metrics = & curl.exe @curlArgs "$BaseUrl$Path"
    $code = $LASTEXITCODE
    [ordered]@{started_utc=$start;finished_utc=[DateTime]::UtcNow.ToString('o');curl_exit=$code;transfer=($metrics | ConvertFrom-Json)} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 "$OutputDirectory/$Name.transfer.json"
    if ($code -ne 0) { throw "$Name curl failed: $code" }
    if (($metrics | ConvertFrom-Json).http_code -ne 200) { throw "$Name HTTP failed" }
}
Invoke-Probe 'health-before' '/health' $null
Invoke-Probe 'models' '/v1/models' $null
$model = ((Get-Content -Raw "$OutputDirectory/models.body" | ConvertFrom-Json).data | Select-Object -First 1).id
foreach ($stream in @($false,$true)) {
    $name = if ($stream) {'chat-stream'} else {'chat-json'}
    $body = @{model=$model;messages=@(@{role='user';content="Acceptance marker $run/$name. Reply with exactly: LAN_OK"});max_tokens=64;reasoning_effort='none';stream=$stream}
    Invoke-Probe $name '/v1/chat/completions' $body
}
$chat = Get-Content -Raw "$OutputDirectory/chat-json.body" | ConvertFrom-Json
if (-not $chat.choices[0].message.content -or $chat.choices[0].finish_reason -ne 'stop') { throw 'JSON completion missing content/stop' }
$sse = Get-Content -Raw "$OutputDirectory/chat-stream.body"
if ($sse -notmatch 'data: \[DONE\]' -or $sse -notmatch 'chat.completion.chunk' -or $sse -notmatch '"finish_reason":"stop"') { throw 'SSE completion incomplete' }
Invoke-Probe 'health-after' '/health' $null
$after = Get-Content -Raw "$OutputDirectory/health-after.body" | ConvertFrom-Json
if (-not $after.model_ready -or $after.fatal_error -or $after.active_requests -ne 0 -or $after.queued_requests -ne 0) { throw 'Runtime not settled/ready' }
[ordered]@{run_id=$run;finished_utc=[DateTime]::UtcNow.ToString('o');http_probes='PASS';json_content=$chat.choices[0].message.content;sse_done=$true;runtime_ready=$after.model_ready;active_requests=$after.active_requests;queued_requests=$after.queued_requests;mac_operator_correlation='pending'} | ConvertTo-Json | Set-Content -Encoding UTF8 "$OutputDirectory/result.json"
Get-Content "$OutputDirectory/result.json"
