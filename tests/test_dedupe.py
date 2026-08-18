from podauto.dedupe import concept_tokens, dedupe_key, normalize_text


def test_word_order_does_not_change_the_key():
    a = dedupe_key("Fishing", "Fishing Legend Angler")
    b = dedupe_key("Fishing", "Angler Legend Fishing")
    assert a == b


def test_pod_boilerplate_is_stripped():
    a = dedupe_key("Fishing", "Funny Vintage Fishing Legend T-Shirt")
    b = dedupe_key("Fishing", "Fishing Legend")
    assert a == b


def test_leetspeak_normalises():
    assert normalize_text("D1sn3y") == "disney"
    assert normalize_text("C0ca-C0la") == "coca cola"


def test_accents_and_punctuation_normalise():
    assert normalize_text("Café  Owner's!") == "cafe owner s"


def test_different_concepts_keep_different_keys():
    a = dedupe_key("Fishing", "Fishing Legend")
    b = dedupe_key("Fishing", "Bass Master Tournament")
    assert a != b


def test_niche_is_part_of_the_key():
    a = dedupe_key("Fishing", "Dad Legend")
    b = dedupe_key("Pickleball", "Dad Legend")
    assert a != b


def test_short_and_noise_tokens_are_dropped():
    assert concept_tokens("The a of to Pickleball") == ["pickleball"]
