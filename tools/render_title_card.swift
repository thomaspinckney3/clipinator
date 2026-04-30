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
let size = NSSize(width: width, height: height)
let image = NSImage(size: size)
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

let title = NSString(string: text)
let maxTextSize = NSSize(width: CGFloat(width) * 0.82, height: CGFloat(height) * 0.55)
var fontSize = CGFloat(82)
var textRect = NSRect.zero
var attrs = attributes(fontSize: fontSize)

while fontSize >= 34 {
    attrs = attributes(fontSize: fontSize)
    textRect = title.boundingRect(
        with: maxTextSize,
        options: [.usesLineFragmentOrigin, .usesFontLeading],
        attributes: attrs
    )
    if textRect.width <= maxTextSize.width && textRect.height <= maxTextSize.height {
        break
    }
    fontSize -= 2
}

image.lockFocus()
NSColor.black.setFill()
NSBezierPath(rect: NSRect(origin: .zero, size: size)).fill()

let drawRect = NSRect(
    x: (CGFloat(width) - ceil(textRect.width)) / 2,
    y: (CGFloat(height) - ceil(textRect.height)) / 2,
    width: ceil(textRect.width),
    height: ceil(textRect.height)
)
title.draw(
    with: drawRect,
    options: [.usesLineFragmentOrigin, .usesFontLeading],
    attributes: attrs
)
image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let representation = NSBitmapImageRep(data: tiff),
      let png = representation.representation(using: .png, properties: [:]) else {
    fputs("failed to render image\n", stderr)
    exit(1)
}

do {
    try png.write(to: URL(fileURLWithPath: outputPath))
} catch {
    fputs("failed to write output: \(error)\n", stderr)
    exit(1)
}
