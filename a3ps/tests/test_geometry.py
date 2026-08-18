"""Unit tests for geometry helpers."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from a3ps.common.geometry import (  # noqa: E402
    GroundPlane,
    bbox_centroid,
    ego_corridor,
    iou,
    point_in_polygon,
)


def test_iou_identical_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 = 50 region.
    # inter=50, union=100+100-50=150 -> 1/3.
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - (1.0 / 3.0)) < 1e-9


def test_bbox_centroid():
    assert bbox_centroid([0, 0, 10, 20]) == (5.0, 10.0)


def test_point_in_polygon():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon([5, 5], square) is True
    assert point_in_polygon([15, 5], square) is False


def test_ego_corridor_shape():
    corridor = ego_corridor(width_m=2.0, length_m=30.0)
    assert len(corridor) == 4
    # Actor directly ahead at (0, 15) is inside; far to the side is outside.
    assert point_in_polygon([0.0, 15.0], corridor) is True
    assert point_in_polygon([5.0, 15.0], corridor) is False


# ---- GroundPlane (BEV projection) -----------------------------------------

W, H = 1280, 720


def _ground_plane():
    config = {}  # use defaults from DEFAULT_GROUND_PLANE
    return GroundPlane.from_config(config, W, H)


def test_bottom_center_maps_near_origin():
    gp = _ground_plane()
    # Bottom-center of the image is the midpoint of the two near ground points
    # (-1.8,0) and (1.8,0) -> should map near (0, 0).
    (x, y) = gp.img_to_bev([[0.5 * W, 1.0 * H]])[0]
    assert abs(x) < 0.05, f"x={x}"
    assert abs(y) < 0.05, f"y={y}"


def test_img_bev_img_round_trip_under_1px():
    gp = _ground_plane()
    pts = [[0.5 * W, 1.0 * H], [0.4 * W, 0.8 * H],
           [0.62 * W, 0.7 * H], [0.31 * W, 0.99 * H]]
    back = gp.bev_to_img(gp.img_to_bev(pts))
    for (ox, oy), (bx, by) in zip(pts, back):
        assert abs(ox - bx) < 1.0 and abs(oy - by) < 1.0, (ox, oy, bx, by)


def test_calibration_corners_map_to_ground_points():
    gp = _ground_plane()
    # The 4 calibration image points must map to their ground coords exactly.
    bev = gp.img_to_bev(gp.image_points)
    for (bx, by), (gx, gy) in zip(bev, gp.ground_points):
        assert abs(bx - gx) < 1e-3 and abs(by - gy) < 1e-3


# ---------------------------------------------------------------------------
# corridor lateral span (extrapolating gate for scale-based TTC)
# ---------------------------------------------------------------------------

def _corridor():
    """Ego corridor quad as the pipeline emits it, for a 1280x720 frame."""
    return [[499.2, 720.0], [780.8, 720.0], [665.6, 432.0], [614.4, 432.0]]


def test_lateral_span_matches_quad_inside_its_own_rows():
    from a3ps.common.geometry import corridor_lateral_span
    lo, hi = corridor_lateral_span(_corridor(), 720.0)
    assert abs(lo - 499.2) < 1e-6 and abs(hi - 780.8) < 1e-6
    lo, hi = corridor_lateral_span(_corridor(), 432.0)
    assert abs(lo - 614.4) < 1e-6 and abs(hi - 665.6) < 1e-6


def test_lateral_span_narrows_with_distance():
    from a3ps.common.geometry import corridor_lateral_span
    near = corridor_lateral_span(_corridor(), 700.0)
    far = corridor_lateral_span(_corridor(), 500.0)
    assert (far[1] - far[0]) < (near[1] - near[0])


def test_lateral_span_extrapolates_above_the_quad():
    """The whole point: rows above top_y must still yield a usable span.

    A plain polygon test reports "outside" for every distant actor, which is
    what made the looming gate reject the actors it most needed to see.
    """
    from a3ps.common.geometry import corridor_lateral_span
    lo, hi = corridor_lateral_span(_corridor(), 380.0)   # above the quad's top
    assert hi > lo
    assert hi - lo < 51.2                                 # narrower than at top_y


def test_bbox_overlaps_corridor_gate():
    from a3ps.common.geometry import bbox_overlaps_corridor
    c = _corridor()
    assert bbox_overlaps_corridor([600, 600, 700, 700], c)          # straddles lane
    assert not bbox_overlaps_corridor([0, 600, 100, 700], c)        # far left
    # a box just outside is admitted once padding widens the span
    box = [800, 650, 860, 700]
    assert not bbox_overlaps_corridor(box, c)
    assert bbox_overlaps_corridor(box, c, pad_frac=0.5)


def test_bbox_overlaps_corridor_far_actor_not_auto_rejected():
    from a3ps.common.geometry import bbox_overlaps_corridor
    c = _corridor()
    # small, distant, lane-centred box sitting ABOVE the corridor quad
    assert bbox_overlaps_corridor([632, 360, 648, 380], c)
