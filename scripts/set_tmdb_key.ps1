# Save TMDB_API_KEY permanently for CinéBot / Mr. Cinephile.
# Usage:
#   .\scripts\set_tmdb_key.ps1
#   .\scripts\set_tmdb_key.ps1 -Key "your_api_key"
#   .\scripts\set_tmdb_key.ps1 -Key "your_api_key" -AlsoSetWindowsUserEnv

param(
    [string]$Key = "",
    [switch]$AlsoSetWindowsUserEnv
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvFile = Join-Path $Root ".env"

if (-not $Key) {
    $secure = Read-Host "Paste your TMDB API key" -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $Key = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

$Key = $Key.Trim()
if (-not $Key -or $Key -eq "your_tmdb_api_key_here") {
    Write-Error "No valid API key provided."
}

$content = @"
# TMDB — loaded automatically by movie_data.py / web_chat.py
# Get a key: https://www.themoviedb.org/settings/api
TMDB_API_KEY=$Key
"@

Set-Content -Path $EnvFile -Value $content.TrimEnd() -Encoding utf8
Write-Host "Saved TMDB_API_KEY to $EnvFile (persists across reboots with your project)."

if ($AlsoSetWindowsUserEnv) {
    setx TMDB_API_KEY $Key | Out-Null
    Write-Host "Also set Windows user environment variable TMDB_API_KEY."
    Write-Host "Open a NEW terminal (or restart Cursor) for setx to take effect in other apps."
} else {
    Write-Host ""
    Write-Host "Optional: also set Windows user env (all terminals, even outside this folder):"
    Write-Host "  .\scripts\set_tmdb_key.ps1 -Key `"<hidden>`" -AlsoSetWindowsUserEnv"
}

Write-Host ""
Write-Host "Restart web_chat.py if it is already running."
