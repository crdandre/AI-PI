import unicodedata
import re

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters for LLM processing while preserving mathematical meaning."""
    # Define mathematical symbols to preserve as escape sequences
    math_symbols = {
        '±': '\u00b1',
        '°': '\u00b0',
        '≥': '\u2265',
        '≤': '\u2264',
        # Basic mathematical symbols
        '→': '\u2192',
        '←': '\u2190', 
        '↔': '\u2194',
        '≠': '\u2260',
        '∞': '\u221E',
        '∑': '\u2211',
        '∏': '\u220F',
        '√': '\u221A',
        '∫': '\u222B',
        # Greek letters
        'π': '\u03C0',
        'μ': '\u03BC',
        'α': '\u03B1',
        'β': '\u03B2',
        'γ': '\u03B3',
        'δ': '\u03B4',
        'ε': '\u03B5',
        'θ': '\u03B8',
        'λ': '\u03BB',
        'σ': '\u03C3',
        'τ': '\u03C4',
        'φ': '\u03C6',
        'ω': '\u03C9',
        # Additional mathematical symbols
        'Δ': '\u0394',
        '×': '\u00D7',
        '÷': '\u00F7',
        '≈': '\u2248',
        '≡': '\u2261',
        '∝': '\u221D',
        '∂': '\u2202',
        '∇': '\u2207',
        '∈': '\u2208',
        '∉': '\u2209',
        '∋': '\u220B',
        '∌': '\u220C',
        '∩': '\u2229',
        '∪': '\u222A'
    }
    
    # First convert any direct symbols to escape sequences
    for symbol, escape_seq in math_symbols.items():
        text = text.replace(symbol, escape_seq)
    
    # Then normalize remaining Unicode to ASCII where possible
    normalized = unicodedata.normalize('NFKD', text)
    
    # Keep only non-combining characters, except preserve our escape sequences
    result = []
    i = 0
    while i < len(normalized):
        if normalized[i:i+6].startswith('\\u'):  # Check for escape sequence
            result.append(normalized[i:i+6])
            i += 6
        elif not unicodedata.combining(normalized[i]):
            result.append(normalized[i])
        i += 1
    
    return ''.join(result)

def clean_citations(text: str) -> str:
    """Remove citations and references section from text while preserving readability."""
    text = normalize_unicode(text)
    
    if 'References' in text:
        text = text.split('References')[0]
    text = re.sub(r'\([^()]*?\d{4}[^()]*?\)', '', text)  # Remove parenthetical citations
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)      # Remove numbered citations
    text = re.sub(r'\s+', ' ', text)                      # Remove multiple spaces
    text = re.sub(r'\s+([.,;:])', r'\1', text)           # Fix spacing around punctuation
    
    return text.strip()

def clean_xml_and_b64(text: str) -> str:
    """Remove XML tags, base64 content, and other non-content markup from text."""
    text = normalize_unicode(text)
    
    text = re.sub(r'<[^>]+>.*?</[^>]+>', '', text)       # Remove XML/HTML tags and contents
    text = re.sub(r'<[^>]+/?>', '', text)                # Remove standalone XML/HTML tags
    text = re.sub(r'[A-Za-z0-9+/=\n]{50,}', '', text)    # Remove base64 encoded data
    text = re.sub(r'ADDIN\s+[A-Z.]+(?:\s*\{[^}]*\})?', '', text)  # Remove EndNote tags
    text = re.sub(r'SEQ\s+(?:Table|Figure)\s*\\[^"]*', '', text)
    
    # Clean up citations while preserving simple numeric ones
    citations = re.findall(r'\[[\d\s,]+\]', text)
    text = re.sub(r'\[.*?\]', '', text)
    if citations:
        text = text.strip() + ' ' + ' '.join(citations)
    
    return re.sub(r'\s+', ' ', text).strip()

def clean_markdown(text: str) -> str:
    """Remove bold/italic formatting but preserve headers."""
    text = normalize_unicode(text)
    
    text = re.sub(r'(?<![#])\*\*(.+?)\*\*', r'\1', text)  # Remove bold not preceded by #
    text = re.sub(r'(?<![#])\*(.+?)\*', r'\1', text)      # Remove italic not preceded by #
    
    return text 