import io
import random
import sys
import types

import pytest
from PIL import Image

from tools import design_compare_pixels as px
from tools.design_compare_judge import Finding, finding_id

MIB = 1048576
WHITE, BLACK = (255, 255, 255), (0, 0, 0)


def page(w=200, h=200, colour=WHITE):
    return Image.new("RGB", (w, h), colour)


def block(img, x, y, w, h, colour=BLACK):
    img.paste(colour, (x, y, x + w, y + h))
    return img


def regions(proto, cand=None, **kw):
    return px.compare_pages(cand or page(*proto.size), proto, **kw)


def box(x, y, width, height):
    return {"x": x, "y": y, "width": width, "height": height}


def test_planted_block_sizes_decide_regions():
    assert regions(block(page(), 20, 20, 10, 10)) == [box(20, 20, 10, 10)]
    assert regions(block(page(), 20, 20, 8, 8)) == [box(20, 20, 8, 8)]
    assert regions(block(page(), 20, 20, 7, 7)) == []
    assert regions(block(page(), 20, 20, 7, 40)) == []


def test_channel_threshold_is_more_than_24():
    assert regions(page(colour=(231, 231, 231))) == []
    assert regions(page(colour=(231, 255, 255))) == []
    assert regions(block(page(), 20, 20, 10, 10, (231, 255, 255))) == []
    assert regions(block(page(), 20, 20, 10, 10, (230, 255, 255))) == [box(20, 20, 10, 10)]
    assert regions(block(page(), 20, 20, 10, 10, (255, 255, 230))) == [box(20, 20, 10, 10)]


def test_region_needs_64_original_pixels():
    ring = block(block(page(), 20, 20, 9, 9), 21, 21, 7, 7, WHITE)
    assert regions(ring) == []
    assert regions(block(page(), 20, 20, 8, 8)) != []


def test_connectivity_and_dilation_merge_regions():
    diagonal = block(block(page(), 10, 10, 8, 8), 18, 18, 8, 8)
    assert len(regions(diagonal)) == 1
    gap2 = block(block(page(), 10, 10, 8, 8), 20, 10, 8, 8)
    assert len(regions(gap2)) == 1
    gap3 = block(block(page(), 10, 10, 8, 8), 21, 10, 8, 8)
    assert len(regions(gap3)) == 2


def test_regions_are_ordered_top_to_bottom_then_left_to_right():
    proto = block(block(block(page(), 100, 10, 10, 10), 10, 10, 10, 10), 50, 60, 10, 10)
    assert regions(proto) == [box(10, 10, 10, 10), box(100, 10, 10, 10), box(50, 60, 10, 10)]


def test_masks_exclude_regions():
    proto = block(block(page(), 20, 20, 10, 10), 100, 100, 10, 10)
    assert regions(proto, masks=[box(15, 15, 30, 30)]) == [box(100, 100, 10, 10)]
    assert regions(proto, masks=[box(0, 0, 200, 200)]) == []
    assert regions(proto, masks=[box(0, 0, 25, 200)]) != []


def test_row_limit_compares_only_rows_above():
    proto = block(page(), 20, 100, 10, 10)
    assert regions(proto, row_limit=50) == []
    assert regions(proto, row_limit=150) == [box(20, 100, 10, 10)]


def test_element_pass_ignores_shift_but_sees_change():
    cand = block(page(), 20, 20, 30, 30)
    proto = block(page(), 60, 50, 30, 30)
    assert regions(proto, cand) != []
    assert px.compare_element(cand, proto, box(20, 20, 30, 30), box(60, 50, 30, 30)) == []
    changed = block(proto, 70, 60, 12, 12, (255, 0, 0))
    assert px.compare_element(cand, changed, box(20, 20, 30, 30), box(60, 50, 30, 30)) == [box(30, 30, 12, 12)]


def test_element_pass_compares_common_size():
    cand = block(page(), 20, 20, 30, 30)
    proto = block(page(), 60, 50, 40, 50)
    assert px.compare_element(cand, proto, box(20, 20, 30, 30), box(60, 50, 40, 50)) == []


def test_images_accepted_as_png_bytes_and_unequal_height_is_fine():
    buf = io.BytesIO()
    block(page(200, 150), 20, 20, 10, 10).save(buf, "PNG")
    assert regions(page(200, 200), buf.getvalue()) == [box(20, 20, 10, 10)]
    assert px.compare_pages(page(200, 150), page(200, 200)) == []


def test_refusals():
    with pytest.raises(px.PixelError):
        px.compare_pages(page(200, 8001), page(200, 8001))
    with pytest.raises(px.PixelError):
        px.compare_pages(page(200, 100), page(201, 100))
    with pytest.raises(ValueError):
        px.cut_crops(page(200, 100), page(201, 100), box(0, 0, 10, 10), None, 1, "0123456789abcdef", 1280)
    assert px.compare_pages(page(1280, 8000), page(1280, 8000)) == []


def test_attach_regions_to_findings_with_layout_type_or_colour():
    def finding(kind, element, prop):
        return Finding(finding_id("/", 1280, kind, element, prop), kind, "/", 1280, element, prop, "", "", "")

    layout, element = finding("layout", "a", "box.x"), finding("element", "b", "presence")
    boxes = {"a": box(0, 0, 50, 50), "b": box(100, 0, 50, 50)}
    inside, in_b, outside = box(10, 10, 10, 10), box(110, 10, 10, 10), box(40, 40, 20, 20)
    attached, undecided = px.attach_regions([inside, in_b, outside], [layout, element], boxes.get)
    assert attached == {layout.id: [inside]}
    assert undecided == [in_b, outside]


def test_crop_rect_is_union_plus_padding_clipped():
    size = (500, 400)
    assert px.crop_rect(size, box(100, 100, 50, 50), box(120, 130, 60, 70)) == (76, 76, 204, 224)
    assert px.crop_rect(size, box(5, 10, 50, 50), box(0, 0, 10, 10)) == (0, 0, 79, 84)
    assert px.crop_rect(size, box(480, 380, 50, 50), box(480, 380, 50, 50)) == (456, 356, 500, 400)


def test_crop_rect_for_missing_or_extra_element_uses_the_side_that_exists():
    size = (500, 400)
    assert px.crop_rect(size, None, box(100, 100, 50, 50)) == (76, 76, 174, 174)
    assert px.crop_rect(size, box(100, 100, 50, 50), None) == px.crop_rect(size, None, box(100, 100, 50, 50))


def png_size(data):
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG"
    return img.size


def test_crops_are_cut_at_the_same_rectangle_and_not_scaled_when_small():
    cand, proto = block(page(), 50, 50, 20, 20), page()
    crops = px.cut_crops(cand, proto, box(50, 50, 20, 20), box(50, 50, 20, 20), 2, "0123456789abcdef", 1280)
    assert not crops.scaled
    assert png_size(crops.candidate[1]) == png_size(crops.prototype[1]) == (68, 68)
    assert Image.open(io.BytesIO(crops.candidate[1])).getpixel((24, 24)) == BLACK
    assert Image.open(io.BytesIO(crops.prototype[1])).getpixel((24, 24)) == WHITE


def test_largest_crop_is_scaled_down_keeping_aspect_ratio():
    cand, proto = page(2000, 1500), page(2000, 1500, (200, 200, 200))
    crops = px.cut_crops(cand, proto, box(0, 0, 2000, 1500), None, 1, "0123456789abcdef", 375)
    assert crops.scaled
    w, h = png_size(crops.candidate[1])
    assert png_size(crops.prototype[1]) == (w, h)
    assert max(w, h) <= 1200
    assert abs(w / h - 2000 / 1500) < 0.01
    assert max(w, h) >= 1200


def test_oversized_png_shrinks_to_fit_and_pair_keeps_one_size():
    noisy = Image.frombytes("RGB", (1200, 1200), random.Random(7).randbytes(1200 * 1200 * 3))
    crops = px.cut_crops(page(1200, 1200, (10, 20, 30)), noisy, box(0, 0, 1200, 1200), None, 1, "0123456789abcdef", 1280)
    assert len(crops.candidate[1]) <= MIB and len(crops.prototype[1]) <= MIB
    assert png_size(crops.candidate[1]) == png_size(crops.prototype[1])
    assert crops.scaled


RED = (255, 0, 0)
FID = "0123456789abcdef"


def decode(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


def count(img, colour):
    return sum(1 for p in img.getdata() if p == colour)


def test_missing_element_below_the_shorter_page_is_kept_on_the_side_that_has_it():
    cand, proto = page(375, 812), block(page(375, 1200), 30, 1000, 30, 20, RED)
    crops = px.cut_crops(cand, proto, None, box(30, 1000, 30, 20), 1, FID, 375)
    c, p = decode(crops.candidate[1]), decode(crops.prototype[1])
    assert c.size == p.size == (78, 68)
    assert count(p, RED) == 600
    assert count(c, RED) == 0
    assert len(set(c.getdata())) == 1 and c.getpixel((0, 0)) != WHITE


def test_extra_element_below_the_shorter_page_is_kept_on_the_side_that_has_it():
    cand, proto = block(page(375, 1200), 30, 1000, 30, 20, RED), page(375, 812)
    crops = px.cut_crops(cand, proto, box(30, 1000, 30, 20), None, 1, FID, 375)
    c, p = decode(crops.candidate[1]), decode(crops.prototype[1])
    assert c.size == p.size == (78, 68)
    assert count(c, RED) == 600
    assert count(p, RED) == 0
    assert len(set(p.getdata())) == 1 and p.getpixel((0, 0)) != WHITE


def test_box_crossing_the_shorter_page_bottom_keeps_every_element_pixel():
    cand, proto = page(375, 812), block(page(375, 1200), 30, 802, 30, 20, RED)
    crops = px.cut_crops(cand, proto, None, box(30, 802, 30, 20), 1, FID, 375)
    c, p = decode(crops.candidate[1]), decode(crops.prototype[1])
    assert c.size == p.size == (78, 68)
    assert count(p, RED) == 600
    assert c.getpixel((0, 0)) == WHITE and c.getpixel((0, 67)) != WHITE
    assert count(c, RED) == 0


def test_element_pass_excludes_masked_descendant_inside_matched_ancestor():
    cand, proto = block(page(375, 812), 50, 50, 20, 20), page(375, 812)
    whole = box(0, 0, 375, 812)
    assert px.compare_element(cand, proto, whole, whole) == [box(50, 50, 20, 20)]
    assert px.compare_element(cand, proto, whole, whole, candidate_masks=[box(50, 50, 20, 20)]) == []
    assert px.compare_element(cand, proto, whole, whole, candidate_masks=[], prototype_masks=[]) == [box(50, 50, 20, 20)]


def test_element_pass_masks_follow_each_sides_own_crop_with_shifted_boxes():
    cand_box, proto_box = box(20, 20, 150, 150), box(60, 40, 150, 150)
    cand = block(block(page(375, 812), 50, 50, 20, 20), 120, 120, 12, 12, RED)
    proto = page(375, 812)
    assert px.compare_element(cand, proto, cand_box, proto_box) == [box(50, 50, 20, 20), box(120, 120, 12, 12)]
    assert px.compare_element(cand, proto, cand_box, proto_box,
                              candidate_masks=[box(50, 50, 20, 20)]) == [box(120, 120, 12, 12)]
    proto_only = block(block(page(375, 812), 90, 70, 20, 20), 160, 140, 12, 12, RED)
    assert px.compare_element(page(375, 812), proto_only, cand_box, proto_box,
                              prototype_masks=[box(90, 70, 20, 20)]) == [box(120, 120, 12, 12)]
    a_only, b_only = block(page(375, 812), 50, 50, 20, 20), block(page(375, 812), 130, 110, 20, 20)
    assert len(px.compare_element(a_only, b_only, cand_box, proto_box)) == 2
    assert px.compare_element(a_only, b_only, cand_box, proto_box,
                              candidate_masks=[box(50, 50, 20, 20)], prototype_masks=[box(130, 110, 20, 20)]) == []


def test_element_pass_masks_keep_their_source_image_ownership():
    cand_box, proto_box = box(20, 20, 150, 150), box(60, 40, 150, 150)
    cand_mask, proto_mask = box(50, 50, 20, 20), box(90, 70, 20, 20)
    cand = block(block(page(375, 812), 50, 50, 20, 20), 90, 70, 12, 12, RED)
    proto = block(page(375, 812), 90, 70, 20, 20)
    assert px.compare_element(cand, proto, cand_box, proto_box,
                              candidate_masks=[cand_mask], prototype_masks=[proto_mask]) == [box(90, 70, 12, 12)]
    assert px.compare_element(proto, cand, proto_box, cand_box,
                              candidate_masks=[proto_mask], prototype_masks=[cand_mask]) == [box(130, 90, 12, 12)]


def test_file_names_follow_the_pattern():
    names = px.crop_names(3, "0123456789abcdef", 375)
    assert names == ("dc-r3-0123456789abcdef-375-candidate.png", "dc-r3-0123456789abcdef-375-prototype.png")
    crops = px.cut_crops(page(), page(), box(0, 0, 10, 10), None, 3, "0123456789abcdef", 375)
    assert (crops.candidate[0], crops.prototype[0]) == names


@pytest.mark.parametrize("fid, width", [("0123456789ABCDEF", 1280), ("0123456789abcde", 1280),
                                        ("0123456789abcdeg", 1280), ("0123456789abcdef", 800)])
def test_bad_id_or_width_refused(fid, width):
    with pytest.raises(px.PixelError):
        px.crop_names(1, fid, width)


def test_module_depends_on_nothing_beyond_stdlib_pillow_and_the_judge():
    allowed = set(sys.stdlib_module_names) | {"PIL"}
    for value in vars(px).values():
        name = value.__name__ if isinstance(value, types.ModuleType) else getattr(value, "__module__", None)
        if name and name != px.__name__:
            assert name.split(".")[0] in allowed or name == "tools.design_compare_judge", name
