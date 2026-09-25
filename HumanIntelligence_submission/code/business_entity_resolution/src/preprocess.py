import re
import unicodedata

# Comprehensive legal entity and company suffixes across US, India, and France
LEGAL_SUFFIXES = {
    # English / US / India
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited',
    'pvt', 'private', 'llp', 'llc', 'co', 'company',
    'enterprises', 'enterprise', 'services', 'service',
    'and', 'sons', 'investments', 'investment', 'holdings', 'holding',
    'group', 'ventures', 'venture', 'associates', 'consulting',
    'solutions', 'industries', 'international', 'technologies', 'tech',
    # French
    'sarl', 'sa', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'scea',
    'gie', 'societe', 'association', 'fondation', 'etablissement',
    # General European
    'gmbh', 'ag', 'bv', 'nv', 'plc',
}

# Country-specific address stop words
ADDR_STOP = {
    # Generic
    'near', 'opp', 'opposite', 'behind', 'null', 'none', 'nan',
    # English road types
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'boulevard', 'blvd',
    'lane', 'ln', 'drive', 'dr', 'way', 'court', 'ct', 'circle', 'cir',
    'trail', 'trl', 'parkway', 'pkwy', 'highway', 'hwy', 'terrace',
    # Building descriptors
    'floor', 'fl', 'block', 'plot', 'unit', 'apt', 'apartment',
    'suite', 'ste', 'building', 'bldg', 'office', 'room',
    # French road types
    'rue', 'chemin', 'impasse', 'place', 'passage', 'allee', 'cours',
    'quai', 'square', 'cite', 'voie', 'bis', 'ter', 'route',
    # Indian location descriptors
    'nagar', 'colony', 'sector', 'phase', 'market', 'bazaar',
    'chowk', 'gali', 'mohalla', 'ward', 'dist', 'district',
    'taluk', 'mandal', 'tehsil', 'main', 'cross', 'layout',
    'extension', 'extn', 'complex', 'enclave', 'vihar',
}


def clean_unicode(text):
    """Normalize text while PRESERVING non-ASCII characters (Tamil, Hindi, French accents, etc.).

    This is critical for matching Indic-script and French records that the old
    ASCII-only cleaner completely destroyed.
    """
    if not isinstance(text, str) or not text:
        return ""
    # NFKC normalization: composing form, keeps Tamil/Hindi/French chars intact
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    # Remove URL patterns but keep the domain words
    text = re.sub(r'https?://', ' ', text)
    text = re.sub(r'www\.', ' ', text)
    text = re.sub(r'\.(com|in|org|net|fr|co|io)(?=\s|$)', ' ', text)
    # Remove punctuation but keep Unicode word characters
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return ' '.join(text.split())


def clean_ascii(text):
    """ASCII-only normalization for inverted index compatibility.

    Used for building token indexes that match across transliterations
    where the Latin-script portion of names/addresses is preserved.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'https?://', ' ', text)
    text = re.sub(r'www\.', ' ', text)
    text = re.sub(r'\.(com|in|org|net|fr|co|io)(?=\s|$)', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())


def extract_core_name(name):
    """Strip legal suffixes to isolate the distinguishing business name tokens."""
    clean = clean_ascii(name)
    if not clean:
        return ""
    words = [w for w in clean.split() if w not in LEGAL_SUFFIXES]
    return ' '.join(words) if words else clean


def extract_tokens(text, is_name=True):
    """Extract significant ASCII tokens (min length 3) for inverted indexing."""
    clean = clean_ascii(text)
    if not clean:
        return set()
    stops = LEGAL_SUFFIXES if is_name else ADDR_STOP
    return {t for t in clean.split() if len(t) >= 3 and t not in stops}


def extract_addr_tokens(address):
    """Extract significant ASCII address tokens for address-based blocking.

    This is the KEY channel for cross-script matching: when Tamil/Hindi
    business names produce empty ASCII representations, the address tokens
    (which are usually in Latin script) provide the matching signal.
    """
    clean = clean_ascii(address)
    if not clean:
        return set()
    return {t for t in clean.split() if len(t) >= 3 and t not in ADDR_STOP}


def extract_digits(address):
    """Extract distinct numerical tokens from address (house/plot numbers, PIN/zip codes).

    Uses permissive pattern without word boundaries to catch digits in
    strings like '45ND' or '6(29)'.
    """
    if not isinstance(address, str) or not address:
        return set()
    return set(re.findall(r'\d{2,}', address))


def has_non_ascii(text):
    """Check if text contains non-ASCII characters (Indic scripts, French accents, etc.)."""
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[^\x00-\x7F]', text))


def char_ngrams(text, n=3):
    """Extract character n-grams for fuzzy Jaccard similarity."""
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}
