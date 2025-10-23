#!/usr/bin/env python3
"""
Script to remove all emojis from the codebase
"""
import re
import os
import glob

def remove_emojis_from_file(file_path):
    """Remove emojis from a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove emojis using regex
        # This pattern matches most emojis including Unicode ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002700-\U000027BF"  # dingbats
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
            "\U0001F018-\U0001F0F5"  # enclosed alphanumeric supplement
            "\U0001F200-\U0001F2FF"  # enclosed CJK letters and months
            "\U0001F300-\U0001F5FF"  # miscellaneous symbols and pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport and map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # geometric shapes extended
            "\U0001F800-\U0001F8FF"  # supplemental arrows-C
            "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
            "\U0001FB00-\U0001FBFF"  # symbols for legacy computing
            "\U0001FC00-\U0001FCFF"  # symbols for legacy computing
            "\U0001FD00-\U0001FDFF"  # symbols for legacy computing
            "\U0001FE00-\U0001FEFF"  # variation selectors
            "\U0001FF00-\U0001FFFF"  # variation selectors
            "\U00002000-\U0000206F"  # general punctuation
            "\U00002070-\U0000209F"  # superscripts and subscripts
            "\U000020A0-\U000020CF"  # currency symbols
            "\U000020D0-\U000020FF"  # combining diacritical marks for symbols
            "\U00002100-\U0000214F"  # letterlike symbols
            "\U00002150-\U0000218F"  # number forms
            "\U00002190-\U000021FF"  # arrows
            "\U00002200-\U000022FF"  # mathematical operators
            "\U00002300-\U000023FF"  # miscellaneous technical
            "\U00002400-\U0000243F"  # control pictures
            "\U00002440-\U0000245F"  # optical character recognition
            "\U00002460-\U000024FF"  # enclosed alphanumerics
            "\U00002500-\U0000257F"  # box drawing
            "\U00002580-\U0000259F"  # block elements
            "\U000025A0-\U000025FF"  # geometric shapes
            "\U00002600-\U0000267F"  # miscellaneous symbols
            "\U00002680-\U0000269F"  # dingbats
            "\U000026A0-\U000026FF"  # miscellaneous symbols
            "\U00002700-\U000027BF"  # dingbats
            "\U000027C0-\U000027EF"  # miscellaneous mathematical symbols-A
            "\U000027F0-\U000027FF"  # supplemental arrows-A
            "\U00002800-\U000028FF"  # braille patterns
            "\U00002900-\U0000297F"  # supplemental arrows-B
            "\U00002980-\U000029FF"  # miscellaneous mathematical symbols-B
            "\U00002A00-\U00002AFF"  # supplemental mathematical operators
            "\U00002B00-\U00002BFF"  # miscellaneous symbols and arrows
            "\U00002C00-\U00002C5F"  # glagolitic
            "\U00002C60-\U00002C7F"  # latin extended-C
            "\U00002C80-\U00002CFF"  # coptic
            "\U00002D00-\U00002D2F"  # georgian supplement
            "\U00002D30-\U00002D7F"  # tifinagh
            "\U00002D80-\U00002DBF"  # ethiopic extended
            "\U00002DC0-\U00002DFF"  # ethiopic extended
            "\U00002E00-\U00002E7F"  # supplemental punctuation
            "\U00002E80-\U00002EFF"  # cjk radicals supplement
            "\U00002F00-\U00002FDF"  # kangxi radicals
            "\U00002FF0-\U00002FFF"  # ideographic description characters
            "\U00003000-\U0000303F"  # cjk symbols and punctuation
            "\U00003040-\U0000309F"  # hiragana
            "\U000030A0-\U000030FF"  # katakana
            "\U00003100-\U0000312F"  # bopomofo
            "\U00003130-\U0000318F"  # hangul compatibility jamo
            "\U00003190-\U0000319F"  # kanbun
            "\U000031A0-\U000031BF"  # bopomofo extended
            "\U000031C0-\U000031EF"  # cjk strokes
            "\U000031F0-\U000031FF"  # katakana phonetic extensions
            "\U00003200-\U000032FF"  # enclosed cjk letters and months
            "\U00003300-\U000033FF"  # cjk compatibility
            "\U00003400-\U00004DBF"  # cjk unified ideographs extension A
            "\U00004DC0-\U00004DFF"  # yijing hexagram symbols
            "\U00004E00-\U00009FFF"  # cjk unified ideographs
            "\U0000A000-\U0000A48F"  # yi syllables
            "\U0000A490-\U0000A4CF"  # yi radicals
            "\U0000A4D0-\U0000A4FF"  # lisu
            "\U0000A500-\U0000A63F"  # vai
            "\U0000A640-\U0000A69F"  # cyrillic extended-B
            "\U0000A6A0-\U0000A6FF"  # bamum
            "\U0000A700-\U0000A71F"  # modifier tone letters
            "\U0000A720-\U0000A7FF"  # latin extended-D
            "\U0000A800-\U0000A82F"  # syloti nagri
            "\U0000A830-\U0000A83F"  # common indic number forms
            "\U0000A840-\U0000A87F"  # phags-pa
            "\U0000A880-\U0000A8DF"  # saurashtra
            "\U0000A8E0-\U0000A8FF"  # devanagari extended
            "\U0000A900-\U0000A92F"  # kayah li
            "\U0000A930-\U0000A95F"  # rejang
            "\U0000A960-\U0000A97F"  # hangul jamo extended-A
            "\U0000A980-\U0000A9DF"  # javanese
            "\U0000A9E0-\U0000A9FF"  # myanmar extended-B
            "\U0000AA00-\U0000AA5F"  # cham
            "\U0000AA60-\U0000AA7F"  # myanmar extended-A
            "\U0000AA80-\U0000AADF"  # tai viet
            "\U0000AAE0-\U0000AAFF"  # meetei mayek extensions
            "\U0000AB00-\U0000AB2F"  # ethiopic extended-A
            "\U0000AB30-\U0000AB6F"  # latin extended-E
            "\U0000AB70-\U0000ABBF"  # cherokee supplement
            "\U0000ABC0-\U0000ABFF"  # meetei mayek
            "\U0000AC00-\U0000D7AF"  # hangul syllables
            "\U0000D7B0-\U0000D7FF"  # hangul jamo extended-B
            "\U0000D800-\U0000DB7F"  # high surrogates
            "\U0000DB80-\U0000DBFF"  # high private use surrogates
            "\U0000DC00-\U0000DFFF"  # low surrogates
            "\U0000E000-\U0000F8FF"  # private use area
            "\U0000F900-\U0000FAFF"  # cjk compatibility ideographs
            "\U0000FB00-\U0000FB4F"  # alphabetic presentation forms
            "\U0000FB50-\U0000FDFF"  # arabic presentation forms-A
            "\U0000FE00-\U0000FE0F"  # variation selectors
            "\U0000FE10-\U0000FE1F"  # vertical forms
            "\U0000FE20-\U0000FE2F"  # combining half marks
            "\U0000FE30-\U0000FE4F"  # cjk compatibility forms
            "\U0000FE50-\U0000FE6F"  # small form variants
            "\U0000FE70-\U0000FEFF"  # arabic presentation forms-B
            "\U0000FF00-\U0000FFEF"  # halfwidth and fullwidth forms
            "\U0000FFF0-\U0000FFFF"  # specials
            "]+"
        )
        
        # Remove emojis and clean up extra spaces
        cleaned_content = emoji_pattern.sub('', content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)  # Clean up multiple spaces
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)  # Clean up multiple newlines
        
        # Only write if content changed
        if cleaned_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"Removed emojis from: {file_path}")
            return True
        else:
            print(f"No emojis found in: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Remove emojis from all files in the project"""
    print("Removing emojis from codebase...")
    
    # Files to process
    files_to_process = [
        "notebooks/training_simulation.py",
        "notebooks/01_training_simulation.ipynb", 
        "src/date_range_checker.py",
        "api_keys_setup.md",
        "llm_configuration_guide.md",
        "codebase_analysis_report.md",
        "BIGQUERY_CACHE_FIX_SUMMARY.md",
        "ALGORITHM_VERIFICATION_REPORT.md",
        "GITHUB_DEPLOYMENT_CHECKLIST.md",
        "README.md",
        "DEPLOYMENT_CHECKLIST.md",
        "DEPLOYMENT_GUIDE.md"
    ]
    
    processed_count = 0
    for file_path in files_to_process:
        if os.path.exists(file_path):
            if remove_emojis_from_file(file_path):
                processed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nProcessed {processed_count} files")
    print("Emoji removal complete!")

if __name__ == "__main__":
    main()
