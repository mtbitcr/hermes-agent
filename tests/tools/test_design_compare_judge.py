import copy
import hashlib

import pytest

from tools import design_compare_judge as j

BASE_STYLE = {
    "font-family": '"Geist", sans-serif', "font-face": "Geist", "font-size": 16,
    "font-weight": 400, "line-height": 24, "color": [0, 0, 0, 1],
    "background-color": None, "border-top-color": None, "border-right-color": None,
    "border-bottom-color": None, "border-left-color": None, "text-decoration-color": None,
    "fill": None, "stroke": None,
}


def el(text="", tag="p", parent="body", box=(0, 0, 100, 20), landmarks=(), role=None,
       label=None, alt=None, fixed=False, style=None):
    x, y, w, h = box
    return {"landmarks": [list(m) for m in landmarks], "tag": tag, "role": role, "label": label,
            "alt": alt, "text": text, "parent": parent, "fixed": fixed,
            "box": {"x": x, "y": y, "width": w, "height": h}, "style": {**BASE_STYLE, **(style or {})}}


def bundle(*elements, width=1280, page="/"):
    return {"schema": "design_compare_bundle_v1", "page": page, "width": width, "elements": list(elements)}


def props(candidate, prototype, allowed=()):
    return sorted(f.property for f in j.compare_bundles(candidate, prototype, allowed).findings)


def keys(b):
    return [f.element for f in j.compare_bundles(b, bundle()).findings]


def test_identical_bundles_have_no_findings():
    b = bundle(el("Hello", tag="h1"), el("Hello", tag="h1"), el("World"))
    assert j.compare_bundles(b, copy.deepcopy(b)).findings == ()


def test_key_is_chain_role_name_and_position():
    b = bundle(
        el("  Résumé  NOW ", tag="a", landmarks=[("main", "A heading " * 8)]),
        el("x", tag="a", landmarks=[("main", None), ("section", "Plans")], role="button", label="Close"),
        el("again", tag="a", landmarks=[("main", None), ("section", "Plans")], role="button", label="Close"),
        el("", tag="div", alt="Logo"),
    )
    heading = ("a heading " * 8)[:40]
    assert keys(b) == [
        f'main:{heading}/a:"résumé now"#1',
        'main/section:plans/button:"close"#1',
        'main/section:plans/button:"close"#2',
        '/div:"logo"#1',
    ]
    assert keys(bundle(el("a" * 80)))[0] == '/p:"' + "a" * 60 + '"#1'


def test_text_pair_is_one_element_finding_with_both_texts():
    (f,) = j.compare_bundles(bundle(el("Buy now", tag="button")), bundle(el("Buy", tag="button"))).findings
    assert (f.kind, f.property, f.measured, f.expected) == ("element", "text", "buy now", "buy")
    assert f.element == '/button:"buy"#1'


def test_text_pair_needs_same_chain_role_and_leftover_position():
    cand = bundle(el("one", tag="a"), el("two", tag="a"), el("x", tag="b"))
    proto = bundle(el("uno", tag="a"), el("dos", tag="a"), el("y", tag="i"))
    found = [(f.property, f.element, f.measured, f.expected) for f in j.compare_bundles(cand, proto).findings]
    assert found == [("text", '/a:"uno"#1', "one", "uno"), ("text", '/a:"dos"#1', "two", "dos"),
                     ("presence", '/i:"y"#1', "absent", "present"), ("presence", '/b:"x"#1', "present", "absent")]


def test_missing_and_extra_are_blocking_element_kind():
    only_proto = j.compare_bundles(bundle(el("a")), bundle(el("a"), el("b", tag="li"))).findings
    only_cand = j.compare_bundles(bundle(el("a"), el("b", tag="li")), bundle(el("a"))).findings
    assert [(f.kind, f.property, f.measured, f.expected) for f in only_proto] == [("element", "presence", "absent", "present")]
    assert [(f.kind, f.property, f.measured, f.expected) for f in only_cand] == [("element", "presence", "present", "absent")]


@pytest.mark.parametrize("index,name", [(0, "box.x"), (1, "box.y"), (2, "box.width"), (3, "box.height")])
def test_box_tolerance_is_two_px(index, name):
    def with_delta(d):
        box = [10, 20, 100, 30]
        box[index] += d
        return bundle(el("a", box=box))
    proto = bundle(el("a", box=(10, 20, 100, 30)))
    assert props(with_delta(2), proto) == [] and props(with_delta(-2), proto) == []
    assert props(with_delta(2.01), proto) == [name] and props(with_delta(-2.01), proto) == [name]


@pytest.mark.parametrize("prop,base,inside,outside", [
    ("font-size", 16, 16.5, 16.51),
    ("line-height", 24, 25, 25.01),
    ("font-weight", 500, 500, 501),
    ("line-height", "normal", "normal", 24),
    ("font-family", '"Geist", sans-serif', "geist, serif", '"Inter", sans-serif'),
    ("font-face", "Geist", "geist", "Arial"),
])
def test_type_tolerances(prop, base, inside, outside):
    proto = bundle(el("a", style={prop: base}))
    assert props(bundle(el("a", style={prop: inside})), proto) == []
    assert props(bundle(el("a", style={prop: outside})), proto) == [prop]
    if prop == "line-height" and base == 24:
        assert props(bundle(el("a", style={prop: 23})), proto) == []
        assert props(bundle(el("a", style={prop: 22.99})), proto) == [prop]


def _family_props(cand_family, proto_family):
    return props(bundle(el("a", style={"font-family": cand_family})),
                 bundle(el("a", style={"font-family": proto_family})))


@pytest.mark.parametrize("cand,proto", [
    ('"ACME, Two", Arial', '"ACME, One", Arial'),
    ("'ACME, Two', Arial", "'ACME, One', Arial"),
    (r'"ACME\", Two", Arial', r'"ACME\", One", Arial'),
    (r"ACME\, Two, Arial", r"ACME\, One, Arial"),
])
def test_font_family_comma_inside_first_family_is_not_a_separator(cand, proto):
    assert _family_props(cand, proto) == ["font-family"]


def test_font_family_same_first_family_ignores_fallbacks():
    assert _family_props('"ACME, One", Arial', "'acme, one', serif, monospace") == []


def test_font_family_identical_quoted_comma_family_has_no_finding():
    assert _family_props('"ACME, One", Arial', '"ACME, One", Arial') == []


def test_colour_tolerance_is_delta_e_two_and_alpha_point_zero_two(monkeypatch):
    proto = bundle(el("a", style={"color": [10, 20, 30, 1]}))
    cand = bundle(el("a", style={"color": [11, 20, 30, 1]}))
    monkeypatch.setattr(j, "ciede2000", lambda a, b: 2.0)
    assert props(cand, proto) == []
    monkeypatch.setattr(j, "ciede2000", lambda a, b: 2.0001)
    assert props(cand, proto) == ["color"]
    monkeypatch.undo()
    assert props(bundle(el("a", style={"color": [10, 20, 30, 0.98]})), proto) == []
    assert props(bundle(el("a", style={"color": [10, 20, 30, 0.97]})), proto) == ["color"]
    assert props(bundle(el("a", style={"fill": [0, 0, 0, 1]})), bundle(el("a"))) == ["fill"]
    assert props(bundle(el("a", style={"color": [0, 0, 0, 1]})), bundle(el("a", style={"color": [255, 255, 255, 1]}))) == ["color"]


def test_constants_match_the_tolerance_table():
    assert (j.BOX_TOLERANCE, j.FONT_SIZE_TOLERANCE, j.LINE_HEIGHT_TOLERANCE, j.COLOUR_DELTA_E, j.ALPHA_TOLERANCE,
            j.PIXEL_CHANNEL_DELTA, j.PIXEL_REGION_MIN_SIDE, j.PIXEL_REGION_MIN_PIXELS) == (2.0, 0.5, 1.0, 2.0, 0.02, 24, 8, 64)


@pytest.mark.parametrize("lab1,lab2,expected", [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
])
def test_ciede2000_matches_sharma_reference_pairs(lab1, lab2, expected):
    assert j.ciede2000(lab1, lab2) == pytest.approx(expected, abs=5e-5)
    assert j.ciede2000(lab2, lab1) == pytest.approx(expected, abs=5e-5)


def test_srgb_white_and_black_in_lab():
    assert j.srgb_to_lab((255, 255, 255)) == pytest.approx((100, 0, 0), abs=0.01)
    assert j.srgb_to_lab((0, 0, 0)) == pytest.approx((0, 0, 0), abs=0.01)


def test_allowed_text_change_lines_are_parsed_and_normalised():
    card = 'Build it.\nAllowed text change: "Buy  NOW" becomes "Purchase"\n  Allowed text change: "a" becomes "b"  \nnot Allowed text change: "x" becomes "y"'
    assert j.parse_allowed_changes(card) == (("buy now", "purchase"), ("a", "b"))


PROTO = bundle(
    el("Buy", tag="button", parent="row", box=(0, 0, 40, 20)),
    el("Next", tag="a", parent="row", box=(50, 0, 50, 20)),
    el("Below", box=(0, 40, 100, 20)),
    el("Elsewhere", parent="other", box=(300, 5, 50, 20)),
)
CAND = bundle(
    el("Purchase", tag="button", parent="row", box=(0, 0, 80, 30)),
    el("Next", tag="a", parent="row", box=(90, 0, 50, 20)),
    el("Below", box=(0, 50, 100, 20)),
    el("Elsewhere", parent="other", box=(300, 5, 50, 20)),
)


def test_allowed_text_change_compensates_x_and_y_shifts():
    allowed = j.parse_allowed_changes('Allowed text change: "Buy" becomes "Purchase"')
    result = j.compare_bundles(CAND, PROTO, allowed)
    assert result.findings == () and result.notes == ()


def test_unlisted_text_change_gets_element_finding_and_uncompensated_shifts():
    found = [(f.kind, f.property, f.element) for f in j.compare_bundles(CAND, PROTO).findings]
    assert sorted(found) == sorted([("element", "text", '/button:"buy"#1'), ("layout", "box.x", '/a:"next"#1'),
                                    ("layout", "box.y", '/p:"below"#1')])


def test_allowed_pair_still_compares_position_and_other_shifts_add_up():
    proto = bundle(el("A", tag="button", parent="r", box=(0, 0, 40, 20)), el("B", tag="button", parent="r", box=(50, 0, 40, 20)),
                   el("End", tag="a", parent="r", box=(100, 0, 10, 20)))
    cand = bundle(el("AA", tag="button", parent="r", box=(0, 0, 50, 20)), el("BB", tag="button", parent="r", box=(60, 5, 70, 20)),
                  el("End", tag="a", parent="r", box=(140, 0, 10, 20)))
    allowed = (("a", "aa"), ("b", "bb"))
    assert props(cand, proto, allowed) == ["box.y"]


def test_listed_change_not_found_is_a_note_not_a_finding():
    b = bundle(el("a"))
    result = j.compare_bundles(b, copy.deepcopy(b), (("gone", "new"),))
    assert result.findings == () and len(result.notes) == 1 and '"gone" becomes "new"' in result.notes[0]


def test_finding_id_is_hash_of_page_width_kind_element_property():
    fid = j.finding_id("/p", 1280, "layout", 'main/p:"a"#1', "box.y")
    assert fid == hashlib.sha256('v1|/p|1280|layout|main/p:"a"#1|box.y'.encode()).hexdigest()[:16]
    assert j.finding_id("/p", 1280, "layout", 'main/p:"a"#1', "box.x") != fid
    assert j.finding_id("/p", 1280, "layout", 'main/p:"b"#1', "box.y") != fid


def test_finding_id_is_stable_when_measured_values_change():
    proto = bundle(el("a", box=(0, 0, 100, 20)))
    first = j.compare_bundles(bundle(el("a", box=(0, 5, 100, 20))), proto).findings
    second = j.compare_bundles(bundle(el("a", box=(0, 9, 100, 20))), proto).findings
    assert [f.id for f in first] == [f.id for f in second] and first[0].measured != second[0].measured
    assert len(first[0].id) == 16
    other = j.compare_bundles(bundle(el("a", box=(7, 0, 100, 20))), proto).findings
    assert other[0].id != first[0].id
    assert (first[0].page, first[0].width, first[0].tolerance) == ("/", 1280, "plus or minus 2 px")


def test_unknown_schema_missing_field_and_mixed_widths_are_refused():
    good = bundle(el("a"))
    with pytest.raises(j.BundleError, match="schema"):
        j.compare_bundles({**good, "schema": "design_compare_bundle_v2"}, good)
    with pytest.raises(j.BundleError, match="elements"):
        j.compare_bundles({k: v for k, v in good.items() if k != "elements"}, good)
    no_box = el("a")
    del no_box["box"]
    with pytest.raises(j.BundleError, match="box"):
        j.compare_bundles(bundle(no_box), good)
    no_weight = el("a")
    del no_weight["style"]["font-weight"]
    with pytest.raises(j.BundleError, match="font-weight"):
        j.compare_bundles(good, bundle(no_weight))
    with pytest.raises(j.BundleError, match="width"):
        j.compare_bundles(bundle(el("a"), width=375), good)
    with pytest.raises(j.BundleError, match="width"):
        j.compare_bundles(bundle(el("a"), width=800), bundle(el("a"), width=800))
    with pytest.raises(j.BundleError, match="page"):
        j.compare_bundles(bundle(el("a"), page="/x"), good)
