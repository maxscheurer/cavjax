import numpy as np
import pytest

from cavjax.quadrature import frequency_for_target, frequency_mesh, vertex_count


def test_target_selection_and_topology():
    assert frequency_for_target(6000) == 24
    assert vertex_count(24) == 5762
    mesh = frequency_mesh(5)
    assert len(mesh.vertices) == 252
    assert len(mesh.faces) == 500
    np.testing.assert_allclose(np.linalg.norm(mesh.directions, axis=1), 1, atol=1e-14)
    assert np.all(mesh.weights > 0)
    np.testing.assert_allclose(mesh.weights.sum(), 4 * np.pi, atol=1e-13)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_target(value):
    with pytest.raises(ValueError):
        frequency_for_target(value)
