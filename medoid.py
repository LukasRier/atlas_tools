"""Simple medoid for 3D coordinates
"""
import numpy as np
from timeit import timeit


def calculate_distance_matrix(points):
    """Get all to all distances

    Args:
        points (numpy.ndarray): Array of shape (n_points, n_axes) representing points in space.

    Returns:
        numpy.ndarray: Array of shape (n_points, n_points) representing all-to-all distances.
    """
    n_points = points.shape[0]

    distance_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            distance_matrix[i, j] = np.linalg.norm(points[i] - points[j])

    return distance_matrix


def fast_distance_matrix(points):
    assert isinstance(points, np.ndarray)

    x_dot_x = np.sum(points**2, axis=1)
    x_dot_y = np.matmul(points, points.T)
    diffs2 = x_dot_x.reshape(-1, 1) - 2 * x_dot_y + x_dot_x.reshape(1, -1)

    distance_squared_matrix = np.abs(diffs2)
    return distance_squared_matrix


def get_medoid(points):
    """Compute medoid for cluster of points.

    Args:
        points (numpy.ndarray): Array of shape (n_points, n_axes) representing points in space.

    Returns:
        (int, numpy.ndarray): integer representing medoid index in points and array representing medoid coordinates.
    """
    distance_matrix = fast_distance_matrix(points)

    total_distances = np.sum(distance_matrix, axis=1)
    medoid_index = np.argmin(total_distances)
    medoid = points[medoid_index]

    return medoid_index, medoid


if __name__ == "__main__":
    test_data = np.random.randn(300, 3)

    dist = calculate_distance_matrix(test_data)
    dist_fast = fast_distance_matrix(test_data)
    print(dist**2 - dist_fast)
    isequal = np.allclose(dist**2, dist_fast)
    # dist_fast[np.isclose(dist,dist_fast)==False]
    print(isequal)

    iterations = 50
    total_time = timeit(
        "calculate_distance_matrix(test_data)", number=iterations, globals=globals()
    )
    print(f"Average time is {total_time / iterations:.5f} seconds")

    total_time = timeit(
        "fast_distance_matrix(test_data)", number=iterations, globals=globals()
    )
    print(f"Average time(fast) is {total_time / iterations:.5f} seconds")
