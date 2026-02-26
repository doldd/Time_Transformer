import jax.numpy as jnp
import jax
import pytest
from src.subspace_curve import segmented_bezier_coeff_fn, bezier_coeff_fn

def test_segmented_bezier_coeff_fn():
    # Test: 6 control points, quadratic segments (degree=2) => 3 segments, t in [0,3]
    coeff_fn = segmented_bezier_coeff_fn(6, 2)

    # Test at various t values
    t_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected_coeffs = [
        jnp.array([1., 0., 0., 0., 0., 0.]),
        jnp.array([0.25, 0.5, 0.25, 0., 0., 0.]),
        jnp.array([0., 0., 1., 0., 0., 0.]),
        jnp.array([0., 0., 0.25, 0.5, 0.25, 0.]),
        jnp.array([0., 0., 0., 0., 1., 0.]),
        jnp.array([0.25, 0., 0., 0., 0.25, 0.5]),
        jnp.array([1., 0., 0., 0., 0, 0.]),
    ]

    for t, expected_c in zip(t_values, expected_coeffs):
        c = coeff_fn(jnp.array(t))
        assert jnp.allclose(c, expected_c), f"t={t}: Expected {expected_c}, got {c}"
        assert jnp.isclose(c.sum(), 1.0), f"t={t}: Coefficients sum to {c.sum()}, expected 1.0"

def test_segmented_bezier_coeff_fn_v2():
    # Test: 5 control points, quadratic segments (degree=2) => 2 segments (open), t in [0,2]
    coeff_fn = segmented_bezier_coeff_fn(5, 2)

    # Test at various t values
    t_values = [0.0, 0.5, 1.0, 1.5, 2.0,]
    expected_coeffs = [
        jnp.array([1., 0., 0., 0., 0.,]),
        jnp.array([0.25, 0.5, 0.25, 0., 0.,]),
        jnp.array([0., 0., 1., 0., 0.,]),
        jnp.array([0., 0., 0.25, 0.5, 0.25,]),
        jnp.array([0., 0., 0., 0., 1.,]),
    ]

    for t, expected_c in zip(t_values, expected_coeffs):
        c = coeff_fn(jnp.array(t))
        assert jnp.allclose(c, expected_c), f"t={t}: Expected {expected_c}, got {c}"
        assert jnp.isclose(c.sum(), 1.0), f"t={t}: Coefficients sum to {c.sum()}, expected 1.0"


def test_segmented_bezier_coeff_fn_independent_connected():
    # Test: 6 control points, quadratic segments (degree=2) => 3 segments, t in [0,3]
    coeff_fn = segmented_bezier_coeff_fn(6, 2, indepent_connected=True)

    # Test at various t values
    t_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected_coeffs = [
        jnp.array([1., 0., 0., 0., 0., 0.]),   # don't touch CP2 for seg 0, only CP0-1 
        jnp.array([0.25, 0.5, 0., 0., 0., 0.]), # don't touch CP2 for seg 0, only CP0-1
        jnp.array([0., 0., 1., 0., 0., 0.]),  # don't touch CP4 for seg 1, only CP2-3
        jnp.array([0., 0., 0.25, 0.5, 0., 0.]),  # don't touch CP4 for seg 1, only CP2-3
        jnp.array([0., 0., 0., 0., 1., 0.]), # don't touch CP4 for seg 2, only CP5-6
        jnp.array([0., 0., 0., 0., 0.25, 0.5]),  # don't touch CP4 for seg 2, only CP5-6
        jnp.array([0., 0., 0., 0., 0, 0.]),  # don't touch CP4 for seg 2, only CP5-CP6 => all zeros since local coeffs at t=0 are [1,0,0]
    ]

    for t, expected_c in zip(t_values, expected_coeffs):
        c = coeff_fn(jnp.array(t))
        assert jnp.allclose(c, expected_c), f"t={t}: Expected {expected_c}, got {c}"


def test_segmented_bezier_coeff_fn_independent_connected_v2():
    # Test: 5 control points, quadratic segments (degree=2) => 2 segments (open), t in [0,2]
    coeff_fn = segmented_bezier_coeff_fn(5, 2, indepent_connected=True)

    # Test at various t values
    t_values = [0.0, 0.5, 1.0, 1.5, 2.0,]
    expected_coeffs = [
        jnp.array([1., 0., 0., 0., 0.,]),  # don't touch CP2 for seg 0, only CP0-1 
        jnp.array([0.25, 0.5, 0., 0., 0.,]),  # don't touch CP2 for seg 0, only CP0-1 
        jnp.array([0., 0., 1., 0., 0.,]),  # seg 1
        jnp.array([0., 0., 0.25, 0.5, 0.25,]), # seg 1
        jnp.array([0., 0., 0., 0., 1.,]), # seg 1
    ]

    for t, expected_c in zip(t_values, expected_coeffs):
        c = coeff_fn(jnp.array(t))
        assert jnp.allclose(c, expected_c), f"t={t}: Expected {expected_c}, got {c}"
        # assert jnp.isclose(c.sum(), 1.0), f"t={t}: Coefficients sum to {c.sum()}, expected 1.0"

def test_segmented_bezier_coeff_fn_gradients():
    # 5 CPs, degree 2 => open, 2 segments, t in [0,2]
    # Seg 0: CPs [0,1,2], Seg 1: CPs [2,3,4]
    coeff_fn = segmented_bezier_coeff_fn(5, 2)
    def curve_point(t, cps):
        c = coeff_fn(t)
        return jnp.dot(c, cps)

    cps = jnp.array([0.0, 1.0, 3.0, 2.0, 5.0])
    # Let me just check gradients are finite/nonzero and flow correctly
    for t_val in [0., 0.5, 1., 1.5, 2.]:
        t = jnp.array(t_val)
        grad = jax.grad(curve_point)(t, cps)
        assert jnp.isfinite(grad), f"t={t_val}: grad is not finite: {grad}"
        assert grad != 0.0, f"t={t_val}: grad is zero"

    coeff_fn = segmented_bezier_coeff_fn(6, 2)
    cps = jnp.array([0.0, 1.0, 3.0, 2.0, 5.0, 6.])
    # Let me just check gradients are finite/nonzero and flow correctly
    for t_val in [0., 0.5, 1., 1.5, 2., 2.5, 3.]:
        t = jnp.array(t_val)
        grad = jax.grad(curve_point)(t, cps)
        assert jnp.isfinite(grad), f"t={t_val}: grad is not finite: {grad}"
        assert grad != 0.0, f"t={t_val}: grad is zero"


def test_segmented_bezier_coeff_fn_endpoints():
    # Open: 5 CPs, degree 2 => 2 segments, t in [0,2]
    coeff_fn = segmented_bezier_coeff_fn(5, 2)
    c0 = coeff_fn(jnp.array(0.0))
    expected_c0 = jnp.array([1., 0., 0., 0., 0.])
    assert jnp.allclose(c0, expected_c0), f"t=0.0: Expected {expected_c0}, got {c0}"

    c_end = coeff_fn(jnp.array(2.0))
    expected_c_end = jnp.array([0., 0., 0., 0., 1.])
    assert jnp.allclose(c_end, expected_c_end), f"t=2.0: Expected {expected_c_end}, got {c_end}"

    # Closed: 6 CPs, degree 2 => 3 segments, t in [0,3], wraps back to CP0
    coeff_fn_closed = segmented_bezier_coeff_fn(6, 2)
    c0 = coeff_fn_closed(jnp.array(0.0))
    expected_c0 = jnp.array([1., 0., 0., 0., 0., 0.])
    assert jnp.allclose(c0, expected_c0), f"closed t=0.0: Expected {expected_c0}, got {c0}"

    c_end = coeff_fn_closed(jnp.array(3.0))
    expected_c_end = jnp.array([1., 0., 0., 0., 0., 0.])  # wraps back to CP0
    assert jnp.allclose(c_end, expected_c_end), f"closed t=3.0: Expected {expected_c_end}, got {c_end}"

def test_segmented_bezier_coeff_fn_num_segments_assertion():
    # num_bends=1 with degree=2: (1-1)%2==0 open → num_segments=0 → fails assertion
    with pytest.raises(AssertionError):
        segmented_bezier_coeff_fn(num_bends=1, segment_degree=2)
    # num_bends=3, degree=4: neither open ((3-1)%4==2) nor closed (3%4==3) → fails assertion
    with pytest.raises(AssertionError):
        segmented_bezier_coeff_fn(num_bends=3, segment_degree=4)


