#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <omp.h>

struct Point {
    double x, y;
    int cluster;
    double minDist;

    Point(double x, double y) : x(x), y(y), cluster(-1), minDist(std::numeric_limits<double>::max()) {}
};

double euclideanDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

void kmeans(std::vector<Point>& points, int k) {
    std::vector<Point> centroids;
    std::srand(std::time(0));

    // Initialize centroids randomly
    for (int i = 0; i < k; ++i) {
        int index = std::rand() % points.size();
        centroids.push_back(points[index]);
    }

    bool changed;
    do {
        changed = false;

        // Assign points to the nearest centroid
        for (auto& point : points) {
            for (int i = 0; i < k; ++i) {
                double dist = euclideanDistance(point, centroids[i]);
                if (dist < point.minDist) {
                    point.minDist = dist;
                    point.cluster = i;
                    changed = true;
                }
            }
        }

        // Update centroids
        std::vector<int> counts(k, 0);
        std::vector<double> sumX(k, 0.0), sumY(k, 0.0);

        for (const auto& point : points) {
            int cluster = point.cluster;
            counts[cluster]++;
            sumX[cluster] += point.x;
            sumY[cluster] += point.y;
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] != 0) {
                centroids[i].x = sumX[i] / counts[i];
                centroids[i].y = sumY[i] / counts[i];
            }
        }

    } while (changed);
}

int main() {
    std::vector<Point> points = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {6.0, 8.0}, {7.0, 9.0}, {8.0, 8.0}
    };
    double runtime;
    int k = 2;
    runtime = omp_get_wtime();
    kmeans(points, k);
    runtime = omp_get_wtime() - runtime;
    for (const auto& point : points) {
        std::cout << "Point (" << point.x << ", " << point.y << ") is in cluster " << point.cluster << std::endl;
    }
    printf("\n runtime: %f", runtime);
    return 0;
}