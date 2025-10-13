# Save as check_dll.ps1
$dll = "C:\Program Files (x86)\OpenFHE\lib\libOPENFHEcore.dll"

if (Test-Path $dll) {
    Write-Host "Checking dependencies for: $dll"
    Write-Host ""

    # Try using dumpbin if available
    $dumpbin = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe"
    $dumpbinPath = Get-Item $dumpbin -ErrorAction SilentlyContinue | Select-Object -First 1

    if ($dumpbinPath) {
        & $dumpbinPath /dependents $dll
    } else {
        Write-Host "dumpbin not found. Install Visual Studio Build Tools or use Dependencies.exe"
        Write-Host "Download: https://github.com/lucasg/Dependencies/releases"
    }
} else {
    Write-Host "DLL not found at: $dll"
}