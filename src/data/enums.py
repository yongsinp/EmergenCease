from enum import StrEnum, auto

__all__ = [
    "Category",
    "Certainty",
    "Event",
    "Language",
    "MsgType",
    "Region",
    "ResponseType",
    "Scope",
    "Severity",
    "Status",
    "Urgency",
]


class Category(StrEnum):
    CBRNE = auto()
    Env = "Environmental"
    Fire = auto()
    Geo = "Geological"
    Health = auto()
    Infra = "Infrastructure"
    Met = "Meteorological"
    Other = auto()
    Rescue = auto()
    Safety = auto()
    Security = auto()
    Transport = auto()


class Event(StrEnum):
    """
    Event types from FCC Alert Templates.
    All event types are in CamelCase, and 911 is replaced with NineOneOne.
    """

    def __eq__(self, other):
        if isinstance(other, str):
            other = other.replace(" ", "")

        """Normalize and compare."""
        if self.name == "NineOneOneOutageAlert":
            return other == "911OutageAlert" or other == self.name

        return super().__eq__(other)

    TornadoEmergency = auto()
    TornadoWarning = auto()
    FlashFloodEmergency = auto()
    FlashFloodWarning = auto()
    SevereThunderstormWarning = auto()
    SnowSquallWarning = auto()
    DustStormWarning = auto()
    HurricaneWarning = auto()
    StormSurgeWarning = auto()
    ExtremeWindWarning = auto()
    TestAlert = auto()
    TsunamiWarning = auto()
    EarthquakeWarning = auto()
    BoilWaterAdvisory = auto()
    NineOneOneOutageAlert = auto()
    AvalancheWarning = auto()
    FireWarning = auto()
    HazardousMaterialsWarning = auto()
    Other = "Other"


class Severity(StrEnum):
    Extreme = auto()
    Minor = auto()
    Moderate = auto()
    Severe = auto()
    Unknown = auto()


class Urgency(StrEnum):
    Expected = auto()
    Future = auto()
    Immediate = auto()
    Past = auto()
    Unknown = auto()


class Certainty(StrEnum):
    Likely = auto()
    Observed = auto()
    Possible = auto()
    Unknown = auto()
    Unlikely = auto()
    VeryLikely = "Very Likely"  # Not part of CAP 1.2, but found in some implementations


class Status(StrEnum):
    Actual = auto()  # Actionable by all targeted recipients
    Exercise = auto()  # Actionable only by designated exercise participants
    System = auto()  # For messages that support alert network internal functions
    Test = auto()


class MsgType(StrEnum):
    Ack = "Acknowledgment"  # Acknowledges receipt and acceptance of the message(s)
    Alert = auto()  # Initial information requiring attention by targeted recipients
    Cancel = auto()  # Cancels the earlier message(s)
    Update = auto()  # Updates and supersedes the earlier message(s)


class ResponseType(StrEnum):
    """
    Response types as per CAP 1.2.
    None is replaced with None_ to avoid conflict with the NoneType.
    """

    def __eq__(self, other):
        """Normalize and compare."""
        if self.name == "None_":
            return other == "None" or other == self.name

        return super().__eq__(other)

    None_ = auto()
    AllClear = auto()  # Not part of CAP 1.2, but found in some implementations
    Assess = auto()
    Avoid = auto()  # Not part of CAP 1.2, but found in some implementations
    Evacuate = auto()
    Execute = auto()
    Monitor = auto()
    Prepare = auto()
    Shelter = auto()


class Scope(StrEnum):
    Private = auto()
    Public = auto()  # For general dissemination to unrestricted audiences.
    Restricted = auto()


class Region(StrEnum):
    """Country codes as per ISO 3166."""
    AD = "Andorra"
    AE = "United Arab Emirates"
    AF = "Afghanistan"
    AG = "Antigua And Barbuda"
    AI = "Anguilla"
    AL = "Albania"
    AM = "Armenia"
    AN = "Netherlands Antilles"
    AO = "Angola"
    AR = "Argentina"
    AS = "American Samoa"
    AT = "Austria"
    AU = "Australia"
    AW = "Aruba"
    AX = "Åland Islands"
    AZ = "Azerbaijan"
    BA = "Bosnia And Herzegovina"
    BB = "Barbados"
    BD = "Bangladesh"
    BE = "Belgium"
    BF = "Burkina Faso"
    BG = "Bulgaria"
    BH = "Bahrain"
    BI = "Burundi"
    BJ = "Benin"
    BM = "Bermuda"
    BN = "Brunei Darussalam"
    BO = "Bolivia"
    BR = "Brazil"
    BS = "Bahamas"
    BT = "Bhutan"
    BW = "Botswana"
    BY = "Belarus"
    BZ = "Belize"
    CA = "Canada"
    CC = "Cocos (Keeling) Islands"
    CD = "Congo, The Democratic Republic Of The"
    CF = "Central African Republic"
    CG = "Congo"
    CH = "Switzerland"
    CI = "Côte D'ivoire"
    CK = "Cook Islands"
    CL = "Chile"
    CM = "Cameroon"
    CN = "China"
    CO = "Colombia"
    CR = "Costa Rica"
    CS = "Serbia And Montenegro"
    CU = "Cuba"
    CV = "Cape Verde"
    CX = "Christmas Island"
    CY = "Cyprus"
    CZ = "Czech Republic"
    DE = "Germany"
    DJ = "Djibouti"
    DK = "Denmark"
    DM = "Dominica"
    DO = "Dominican Republic"
    DZ = "Algeria"
    EC = "Ecuador"
    EE = "Estonia"
    EG = "Egypt"
    ER = "Eritrea"
    ES = "Spain"
    ET = "Ethiopia"
    FI = "Finland"
    FJ = "Fiji"
    FK = "Falkland Islands (Malvinas)"
    FM = "Micronesia, Federated States Of"
    FO = "Faroe Islands"
    FR = "France"
    GA = "Gabon "
    GB = "United Kingdom"
    GD = "Grenada"
    GE = "Georgia"
    GF = "French Guiana"
    GH = "Ghana"
    GI = "Gibraltar"
    GL = "Greenland"
    GM = "Gambia"
    GN = "Guinea"
    GP = "Guadeloupe"
    GQ = "Equatorial Guinea"
    GR = "Greece"
    GT = "Guatemala"
    GU = "Guam"
    GW = "Guinea-Bissau"
    GY = "Guyana"
    HK = "Hong Kong"
    HN = "Honduras"
    HR = "Croatia"
    HT = "Haiti"
    HU = "Hungary"
    ID = "Indonesia"
    IE = "Ireland"
    IL = "Israel"
    IN = "India"
    IO = "British Indian Ocean Territory"
    IQ = "Iraq"
    IR = "Iran, Islamic Republic Of"
    IS = "Iceland"
    IT = "Italy"
    JM = "Jamaica"
    JO = "Jordan"
    JP = "Japan"
    KE = "Kenya"
    KG = "Kyrgyzstan"
    KH = "Cambodia"
    KI = "Kiribati"
    KM = "Comoros"
    KN = "Saint Kitts And Nevis"
    KP = "Korea, Democratic People's Republic Of"
    KR = "Korea, Republic Of"
    KW = "Kuwait"
    KY = "Cayman Islands"
    KZ = "Kazakhstan"
    LA = "Lao People's Democratic Republic"
    LB = "Lebanon"
    LC = "Saint Lucia"
    LI = "Liechtenstein"
    LK = "Sri Lanka"
    LR = "Liberia"
    LS = "Lesotho"
    LT = "Lithuania"
    LU = "Luxembourg"
    LV = "Latvia"
    LY = "Libyan Arab Jamahiriya"
    MA = "Morocco"
    MC = "Monaco"
    MD = "Moldova, Republic Of"
    MG = "Madagascar"
    MH = "Marshall Islands"
    MK = "Macedonia, The Former Yugoslav Republic Of"
    ML = "Mali"
    MM = "Myanmar"
    MN = "Mongolia"
    MO = "Macao"
    MP = "Northern Mariana Islands"
    MQ = "Martinique"
    MR = "Mauritania"
    MS = "Montserrat"
    MT = "Malta"
    MU = "Mauritius"
    MV = "Maldives"
    MW = "Malawi"
    MX = "Mexico"
    MY = "Malaysia"
    MZ = "Mozambique"
    NA = "Namibia"
    NC = "New Caledonia"
    NE = "Niger"
    NF = "Norfolk Island"
    NG = "Nigeria"
    NI = "Nicaragua"
    NL = "Netherlands"
    NO = "Norway"
    NP = "Nepal"
    NR = "Nauru"
    NU = "Niue"
    NZ = "New Zealand"
    OM = "Oman"
    PA = "Panama"
    PE = "Peru"
    PF = "French Polynesia"
    PG = "Papua New Guinea"
    PH = "Philippines"
    PK = "Pakistan"
    PL = "Poland"
    PM = "Saint Pierre And Miquelon"
    PN = "Pitcairn"
    PR = "Puerto Rico"
    PS = "Palestinian Territory, Occupied"
    PT = "Portugal"
    PW = "Palau"
    PY = "Paraguay"
    QA = "Qatar"
    RE = "Réunion"
    RO = "Romania"
    RU = "Russian Federation"
    RW = "Rwanda"
    SA = "Saudi Arabia"
    SB = "Solomon Islands"
    SC = "Seychelles"
    SD = "Sudan"
    SE = "Sweden"
    SG = "Singapore"
    SH = "Saint Helena "
    SI = "Slovenia"
    SK = "Slovakia"
    SL = "Sierra Leone"
    SM = "San Marino"
    SN = "Senegal"
    SO = "Somalia"
    SR = "Suriname"
    ST = "Sao Tome And Principe"
    SV = "El Salvador"
    SY = "Syrian Arab Republic"
    SZ = "Swaziland"
    TC = "Turks And Caicos Islands"
    TD = "Chad"
    TG = "Togo"
    TH = "Thailand"
    TJ = "Tajikistan"
    TK = "Tokelau"
    TL = "Timor-Leste"
    TM = "Turkmenistan"
    TN = "Tunisia"
    TO = "Tonga"
    TR = "Turkey"
    TT = "Trinidad And Tobago"
    TV = "Tuvalu"
    TW = "Taiwan, Province Of China"
    TZ = "Tanzania, United Republic Of"
    UA = "Ukraine"
    UG = "Uganda"
    UM = "United States Minor Outlying Islands"
    US = "United States"
    UY = "Uruguay"
    UZ = "Uzbekistan"
    VA = "Holy See (Vatican City State)"
    VC = "Saint Vincent And The Grenadines"
    VE = "Venezuela"
    VG = "Virgin Islands, British"
    VI = "Virgin Islands, U.S."
    VN = "Viet Nam"
    VU = "Vanuatu"
    WF = "Wallis And Futuna"
    WS = "Samoa"
    YE = "Yemen"
    YT = "Mayotte"
    YU = "Yugoslavia"
    ZA = "South Africa"
    ZM = "Zambia"
    ZW = "Zimbabwe"


class Language(StrEnum):
    """
    Undercased language codes as per ISO 639.
    Hyphens are replaced with underscores.
    """

    def __eq__(self, other):
        """Undercase and remove hyphens for comparison."""
        if isinstance(other, str):
            normalized = other.lower().replace("-", "")
            return self.name.replace("_", "") == normalized

        return super().__eq__(other)

    aa = "Afar"
    ab = "Abkhazian"
    ae = "Avestan"
    af = "Afrikaans"
    ak = "Akan"
    am = "Amharic"
    an = "Aragonese"
    ar = "Arabic"
    as_ = "Assamese"
    av = "Avaric"
    ay = "Aymara"
    az = "Azerbaijani"
    ba = "Bashkir"
    be = "Belarusian"
    bg = "Bulgarian"
    bh = "Bihari"
    bi = "Bislama"
    bm = "Bambara"
    bn = "Bengali"
    bo = "Tibetan"
    br = "Breton"
    bs = "Bosnian"
    byn = "Blin"
    ca = "Catalan"
    ce = "Chechen"
    ch = "Chamorro"
    co = "Corsican"
    cr = "Cree"
    cs = "Czech"
    cu = "Church Slavic"
    cv = "Chuvash"
    cy = "Welsh"
    da = "Danish"
    de = "German"
    din = "Dinka"
    dsb = "Lower Sorbian"
    dv = "Divehi"
    dz = "Dzongkha"
    ee = "Ewe"
    el = "Greek"
    en = "English"
    eo = "Esperanto"
    es = "Spanish"
    et = "Estonian"
    eu = "Basque"
    fa = "Persian"
    ff = "Fulah"
    fi = "Finnish"
    fj = "Fijian"
    fo = "Faroese"
    fr = "French"
    fy = "Frisian"
    ga = "Irish"
    gd = "Gaelic"
    gez = "Geez"
    gil = "Gilbertese"
    gl = "Galician"
    gn = "Guarani"
    gu = "Gujarati"
    gv = "Manx"
    ha = "Hausa"
    hak = "Hakka"
    haw = "Hawaiian"
    he = "Hebrew"
    hi = "Hindi"
    ho = "Hiri Motu"
    hr = "Croatian"
    hsb = "Upper Sorbian"
    ht = "Haitian"
    hu = "Hungarian"
    hy = "Armenian"
    hz = "Herero"
    ia = "Interlingua"
    id = "Indonesian"
    ie = "Interlingue"
    ig = "Igbo"
    ii = "Sichuan Yi"
    ik = "Inupiaq"
    io = "Ido"
    is_ = "Icelandic"
    it = "Italian"
    iu = "Inuktitut"
    ja = "Japanese"
    jv = "Javanese"
    ka = "Georgian"
    kg = "Kongo"
    ki = "Kikuyu"
    kj = "Kuanyama"
    kk = "Kazakh"
    kl = "Kalaallisut"
    km = "Khmer"
    kn = "Kannada"
    ko = "Korean"
    kok = "Konkani"
    kr = "Kanuri"
    ks = "Kashmiri"
    ku = "Kurdish"
    kv = "Komi"
    kw = "Cornish"
    ky = "Kirghiz"
    la = "Latin"
    lb = "Luxembourgish"
    lg = "Ganda"
    li = "Limburgan"
    ln = "Lingala"
    lo = "Lao"
    lt = "Lithuanian"
    lu = "Luba-Katanga"
    lv = "Latvian"
    mg = "Malagasy"
    mh = "Marshallese"
    mi = "Maori"
    mk = "Macedonian"
    ml = "Malayalam"
    mn = "Mongolian"
    mo = "Moldavian"
    mr = "Marathi"
    ms = "Malay"
    mt = "Maltese"
    my = "Burmese"
    na = "Nauru"
    nb = "Bokmål"
    nd = "N. Ndebele"
    nds = "Low German"
    ne = "Nepali"
    ng = "Ndonga"
    nl = "Dutch"
    nn = "Nynorsk"
    no = "Norwegian"
    nr = "S. Ndebele"
    nv = "Navajo"
    ny = "Chichewa"
    oc = "Occitan"
    oj = "Ojibwa"
    om = "Oromo"
    or_ = "Oriya"
    os = "Ossetian"
    pa = "Panjabi"
    pi = "Pali"
    pl = "Polish"
    ps = "Pushto"
    pt = "Portuguese"
    qu = "Quechua"
    rm = "Raeto-Romance"
    rn = "Rundi"
    ro = "Romanian"
    ru = "Russian"
    rw = "Kinyarwanda"
    sa = "Sanskrit"
    sc = "Sardinian"
    sd = "Sindhi"
    se = "Northern Sami"
    sg = "Sango"
    si = "Sinhala"
    sid = "Sidamo"
    sk = "Slovak"
    sm = "Samoan"
    sma = "S. Sami"
    sme = "N. Sami"
    smn = "Inari Sami"
    sn = "Shona"
    so = "Somali"
    sq = "Albanian"
    sr = "Serbian"
    ss = "Swati"
    st = "Southern Sotho"
    su = "Sundanese"
    sv = "Swedish"
    sw = "Swahili"
    syr = "Syriac"
    ta = "Tamil"
    te = "Telugu"
    tg = "Tajik"
    th = "Thai"
    ti = "Tigrinya"
    tig = "Tigre"
    tk = "Turkmen"
    tl = "Tagalog"
    tn = "Tswana"
    to = "Tongan"
    tr = "Turkish"
    ts = "Tsonga"
    tt = "Tatar"
    tvl = "Tuvalu"
    tw = "Twi"
    ty = "Tahitian"
    ug = "Uighur"
    uk = "Ukrainian"
    ur = "Urdu"
    uz = "Uzbek"
    ve = "Venda"
    vi = "Vietnamese"
    vo = "Volapük"
    wa = "Walloon"
    wal = "Walamo"
    wen = "Sorbian"
    wo = "Wolof"
    xh = "Xhosa"
    yi = "Yiddish"
    yo = "Yoruba"
    za = "Zhuang"
    zh = "Chinese"
    zu = "Zulu"
