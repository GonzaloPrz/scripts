import re
import sys
from pathlib import Path

def parse_textgrid(textgrid_file):
    with open(textgrid_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    intervals = []
    xmin, xmax, text = None, None, None
    for i, line in enumerate(lines):
        if 'phones' in line:
            break
        if 'xmin =' in line:
            xmin = float(re.search(r'xmin = ([0-9\.]+)', line).group(1))
        elif 'xmax =' in line:
            xmax = float(re.search(r'xmax = ([0-9\.]+)', line).group(1))
        elif 'text =' in line:
            text = re.search(r'text = "(.*)"', line).group(1)
            intervals.append((xmin, xmax, text))
    
    return intervals

def save_audacity_format(intervals, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for xmin, xmax, text in intervals:
            if text.strip():  # Avoid empty labels
                f.write(f"{xmin}\t{xmax}\t{text}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_textgrids.py input output")
        sys.exit(1)
    
    textgrid_folder = sys.argv[1]
    output_folder = sys.argv[2]
    for textgrid_file in Path(textgrid_folder).glob("*.TextGrid"):
        output_file = Path(output_folder,textgrid_file.stem + '.txt')
        
        intervals = parse_textgrid(textgrid_file)
        save_audacity_format(intervals, output_file)
        print(f"Converted {textgrid_file} to Audacity label format: {output_file}")

if __name__ == "__main__":
    main()
