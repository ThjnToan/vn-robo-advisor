$env:PATH = "C:\Program Files\nodejs;" + $env:PATH
Write-Host "Node: $(node -v)"
Write-Host "NPM: $(npm -v)"
Write-Host "Running create-next-app..."
$proc = Start-Process -FilePath "C:\Program Files\nodejs\node.exe" `
    -ArgumentList ('"C:\Program Files\nodejs\node_modules\npm\bin\npx-cli.js"', '-y', 'create-next-app@latest', 'frontend', '--ts', '--tailwind', '--eslint', '--app', '--src-dir', '--import-alias', '@/*', '--use-npm') `
    -NoNewWindow -Wait -PassThru
Write-Host "Exit code: $($proc.ExitCode)"
