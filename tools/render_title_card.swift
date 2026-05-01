import AppKit
import Foundation

if CommandLine.arguments.count < 5 {
    fputs("usage: render_title_card.swift OUTPUT WIDTH HEIGHT TEXT\n", stderr)
    exit(2)
}

let outputPath = CommandLine.arguments[1]
guard let width = Int(CommandLine.arguments[2]),
      let height = Int(CommandLine.arguments[3]) else {
    fputs("invalid width or height\n", stderr)
    exit(2)
}

let text = CommandLine.arguments.dropFirst(4).joined(separator: " ")
let paragraph = NSMutableParagraphStyle()
paragraph.alignment = .center
paragraph.lineBreakMode = .byWordWrapping
paragraph.lineSpacing = 8

func attributes(fontSize: CGFloat) -> [NSAttributedString.Key: Any] {
    [
        .font: NSFont.systemFont(ofSize: fontSize, weight: .bold),
        .foregroundColor: NSColor.white,
        .paragraphStyle: paragraph,
    ]
}

let maxTextSize = NSSize(width: CGFloat(width) * 0.72, height: CGFloat(height) * 0.55)
var fontSize = CGFloat(92)
var attrs = attributes(fontSize: fontSize)
var wrappedLines: [String] = []
var lineHeight = CGFloat(0)
var totalTextHeight = CGFloat(0)

func measuredWidth(_ text: String, attrs: [NSAttributedString.Key: Any]) -> CGFloat {
    NSString(string: text).size(withAttributes: attrs).width
}

func measuredHeight(_ attrs: [NSAttributedString.Key: Any]) -> CGFloat {
    NSString(string: "Ag").size(withAttributes: attrs).height
}

func wrapLines(_ text: String, attrs: [NSAttributedString.Key: Any], maxWidth: CGFloat) -> [String] {
    let words = text.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    var lines: [String] = []
    var current = ""
    for word in words {
        let candidate = current.isEmpty ? word : "\(current) \(word)"
        if !current.isEmpty && measuredWidth(candidate, attrs: attrs) > maxWidth {
            lines.append(current)
            current = word
        } else {
            current = candidate
        }
    }
    if !current.isEmpty {
        lines.append(current)
    }
    return lines.isEmpty ? [text] : lines
}

while fontSize >= 34 {
    attrs = attributes(fontSize: fontSize)
    wrappedLines = wrapLines(text, attrs: attrs, maxWidth: maxTextSize.width)
    lineHeight = measuredHeight(attrs)
    totalTextHeight = CGFloat(wrappedLines.count) * lineHeight + CGFloat(max(0, wrappedLines.count - 1)) * paragraph.lineSpacing
    let maxLineWidth = wrappedLines.map { measuredWidth($0, attrs: attrs) }.max() ?? 0
    if maxLineWidth <= maxTextSize.width && totalTextHeight <= maxTextSize.height {
        break
    }
    fontSize -= 2
}

guard let representation = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: width,
    pixelsHigh: height,
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
) else {
    fputs("failed to allocate bitmap\n", stderr)
    exit(1)
}
representation.size = NSSize(width: width, height: height)

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: representation)
NSColor.black.setFill()
NSBezierPath(rect: NSRect(x: 0, y: 0, width: width, height: height)).fill()

var y = (CGFloat(height) - totalTextHeight) / 2
for line in wrappedLines.reversed() {
    let lineWidth = measuredWidth(line, attrs: attrs)
    let drawRect = NSRect(
        x: (CGFloat(width) - ceil(lineWidth)) / 2,
        y: y,
        width: ceil(lineWidth),
        height: ceil(lineHeight)
    )
    NSString(string: line).draw(
        with: drawRect,
        options: [.usesLineFragmentOrigin, .usesFontLeading],
        attributes: attrs
    )
    y += lineHeight + paragraph.lineSpacing
}
NSGraphicsContext.restoreGraphicsState()

guard let png = representation.representation(using: .png, properties: [:]) else {
    fputs("failed to render image\n", stderr)
    exit(1)
}

do {
    try png.write(to: URL(fileURLWithPath: outputPath))
} catch {
    fputs("failed to write output: \(error)\n", stderr)
    exit(1)
}
