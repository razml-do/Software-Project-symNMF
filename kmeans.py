import sys
import math

EPSILON = 1e-4

def euclidean_distance(p, q):
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def centroids_converged(prev, current, epsilon):
    return all(euclidean_distance(p, c) < epsilon for p, c in zip(prev, current))

def print_centroids(centroids):
    for point in centroids:
        print(",".join(f"{coord:.4f}" for coord in point))

def main():
    if len(sys.argv) not in (2, 3):
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    iter_max = 300
    if len(sys.argv) == 3:
        try:
            iter_max = int(sys.argv[2])
        except ValueError:
            print("An Error Has Occurred")
            sys.exit(1)

    if iter_max <= 1 or iter_max >= 1000:
        print("Incorrect maximum iteration!")
        sys.exit(1)

    data = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            try:
                point = [float(x) for x in line.split(',')]
            except ValueError:
                print("An Error Has Occurred")
                sys.exit(1)
            data.append(point)

    if not data:
        print("An Error Has Occurred")
        sys.exit(1)

    N = len(data)
    d = len(data[0])
    if any(len(point) != d for point in data):
        print("An Error Has Occurred")
        sys.exit(1)

    if k <= 1 or k >= N:
        print("Incorrect number of clusters!")
        sys.exit(1)

    centroids = [list(data[i]) for i in range(k)]
    prev_centroids = [[0.0] * d for _ in range(k)]

    for _ in range(iter_max):
        clusters = [[] for _ in range(k)]

        # Assignment step
        for idx, point in enumerate(data):
            distances = [euclidean_distance(point, c) for c in centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(idx)

        # Update step
        for i, cluster in enumerate(clusters):
            if cluster:  # avoid division by zero
                new_centroid = [0.0] * d
                for idx in cluster:
                    for j in range(d):
                        new_centroid[j] += data[idx][j]
                centroids[i] = [val / len(cluster) for val in new_centroid]

        if centroids_converged(prev_centroids, centroids, EPSILON):
            break
        prev_centroids = [list(c) for c in centroids]

    print_centroids(centroids)
    sys.exit(0)

if __name__ == "__main__":
    main()