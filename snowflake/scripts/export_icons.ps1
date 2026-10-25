param(
  [Parameter(Mandatory)] [string] $Template,
  [Parameter(Mandatory)] [string] $OutDir,
  [int[]] $Slides = @(73, 74, 75, 76, 77, 78, 79, 80, 81),
  [int] $Zoom = 10    # render scale: 1 pt = $Zoom px (slide is 720 x 405 pt)
)
# Exports each icon on the template's icon slides as a transparent PNG named after its label,
# by rendering the slide once at high resolution and cropping each icon's bounds. Writes
# index.csv (file, slide, label). Opens the template read-only, with no window.

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force $OutDir | Out-Null
$wasRunning = [bool](Get-Process POWERPNT -ErrorAction SilentlyContinue)
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($Template, -1, 0, 0)   # ReadOnly, titled, no window
$RenderW = [int]($pres.PageSetup.SlideWidth * $Zoom); $RenderH = [int]($pres.PageSetup.SlideHeight * $Zoom)

function Get-Text($s) {
  try { if ($s.HasTextFrame -and $s.TextFrame.HasText) { return ($s.TextFrame.TextRange.Text -replace '\s+', ' ').Trim() } } catch {}
  return ''
}
function Get-Slug($t) { ($t.ToLower() -replace '&amp;', 'and' -replace '[^a-z0-9]+', '-').Trim('-') }

$rows = New-Object System.Collections.Generic.List[object]
$used = @{}
$tmp = Join-Path $env:TEMP "icon-slide-render.png"
try {
foreach ($n in $Slides) {
  $slide = $pres.Slides.Item($n)
  $slide.Export($tmp, "PNG", $RenderW, $RenderH)
  $render = [System.Drawing.Bitmap]::FromFile($tmp)
  $texts = @(); $icons = @()
  foreach ($s in $slide.Shapes) {
    $t = Get-Text $s
    $isGroup = ($s.Type -eq 6)
    if ($t -and -not $isGroup) { if ($t.Length -le 40) { $texts += [pscustomobject]@{ T=$t; L=$s.Left; Tp=$s.Top; W=$s.Width } }; continue }
    if ($s.Width -le 110 -and $s.Height -le 110 -and $s.Width -ge 8 -and $s.Height -ge 8) { $icons += $s }
  }
  foreach ($ic in $icons) {
    $cx = $ic.Left + $ic.Width / 2; $bottom = $ic.Top + $ic.Height
    $lab = $texts | Where-Object { $_.Tp -ge ($bottom - 6) -and $_.Tp -le ($bottom + 45) -and [math]::Abs(($_.L + $_.W / 2) - $cx) -le [math]::Max(30, $_.W / 2) } |
           Sort-Object Tp | Select-Object -First 2
    $label = ($lab | ForEach-Object { $_.T }) -join ' '
    $inner = ''
    if ($ic.Type -eq 6) { foreach ($g in $ic.GroupItems) { $gt = Get-Text $g; if ($gt) { $inner = ($inner + ' ' + $gt).Trim() } } }
    if ($inner) { $label = ($inner + ' ' + $label).Trim() }
    if (-not $label) { $label = "unlabeled-$n-" + [int]$ic.Left + '-' + [int]$ic.Top }
    $slug = Get-Slug $label
    if ($slug.Length -gt 60) { $slug = $slug.Substring(0, 60).Trim('-') }
    if ($used.ContainsKey($slug)) { $used[$slug]++; $slug = "$slug-$($used[$slug])" } else { $used[$slug] = 1 }
    # crop the icon's bounds (+2 pt) from the render; white becomes transparent
    $pad = 2
    $x = [math]::Max(0, [int](($ic.Left - $pad) * $Zoom)); $y = [math]::Max(0, [int](($ic.Top - $pad) * $Zoom))
    $cw = [math]::Min($RenderW - $x, [int](($ic.Width + 2 * $pad) * $Zoom)); $ch = [math]::Min($RenderH - $y, [int](($ic.Height + 2 * $pad) * $Zoom))
    if ($cw -lt 20 -or $ch -lt 20) { continue }   # off the slide
    $crop = $render.Clone((New-Object System.Drawing.Rectangle $x, $y, $cw, $ch), [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $crop.MakeTransparent([System.Drawing.Color]::White)
    $crop.Save((Join-Path $OutDir "$slug.png"), [System.Drawing.Imaging.ImageFormat]::Png); $crop.Dispose()
    $rows.Add([pscustomobject]@{ file = "$slug.png"; slide = $n; label = $label })
  }
  $render.Dispose()
}
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutDir 'index.csv')
} finally {
  if ($render) { try { $render.Dispose() } catch {} }
  Remove-Item $tmp -ErrorAction SilentlyContinue
  try { $pres.Close() } catch {}
  if (-not $wasRunning) { try { $app.Quit() } catch {} }
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
"exported $($rows.Count) icons"
