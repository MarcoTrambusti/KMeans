#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <matplot/matplot.h>

struct Point {
    double x, y;
    int cluster;
    double minDist;
    Point(): x(0.0),y(0.0),cluster(-1),minDist(std::numeric_limits<double>::max()) {}
    Point(double x, double y) : x(x), y(y), cluster(-1), minDist(std::numeric_limits<double>::max()) {}

    [[nodiscard]] double distance(Point p) const {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

/*double euclideanDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}*/

void kmeans(std::vector<Point>& points, int k, int epochs, double tolerance) {
    std::vector<Point> centroids;
    std::srand(std::time(0));

    // Initialize centroids randomly
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points[std::rand() % points.size()]);
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        bool converged = true;

        // Assign points to the nearest centroid
        for (auto& point : points) {
            for (int i = 0; i < k; ++i) {
                double dist = point.distance(centroids[i]);
                if (dist < point.minDist) {
                    point.minDist = dist;
                    point.cluster = i;
                }
            }
        }

        std::vector<int> nPoints(k, 0);
        std::vector<double> sumX(k, 0.0), sumY(k, 0.0);

        // Iterate over points to append data to centroids
        for (auto& point : points) {
            int clusterId = point.cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += point.x;
            sumY[clusterId] += point.y;

            point.minDist = std::numeric_limits<double>::max();  // reset distance
        }

        // Compute the new centroids
        for (int i = 0; i < k; ++i) {
            if (nPoints[i] != 0) { // Check to avoid division by zero
                double newX = sumX[i] / nPoints[i];
                double newY = sumY[i] / nPoints[i];
                if (std::abs(newX - centroids[i].x) > tolerance || std::abs(newY - centroids[i].y) > tolerance) {
                    converged = false;
                }
                centroids[i].x = newX;
                centroids[i].y = newY;
            }
        }

        if (converged) {
            std::cout << "Converged after " << epoch + 1 << " epochs." << std::endl;
            break;
        }
    }
}

void parallel_kmeans(std::vector<Point>& points, int k, int epochs, double tolerance) {
    std::vector<Point> centroids(k);

    // Initialize centroids randomly
    std::srand(std::time(0));
    for (int i = 0; i < k; ++i) {
        centroids[i] = points[std::rand() % points.size()];
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Reset temp buffers for centroids
        std::vector<int> nPoints(k, 0);
        std::vector<double> sumX(k, 0.0), sumY(k, 0.0);

        // Assign points to the nearest centroid
        #pragma omp parallel
        {
            std::vector<int> local_nPoints(k, 0);
            std::vector<double> local_sumX(k, 0.0), local_sumY(k, 0.0);

            #pragma omp for
            for (size_t j = 0; j < points.size(); ++j) {
                auto& point = points[j];
                double minDist = std::numeric_limits<double>::max();
                int bestCluster = -1;

                // Find the nearest centroid
                for (int i = 0; i < k; ++i) {
                    double dist = point.distance(centroids[i]);
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = i;
                    }
                }
                point.cluster = bestCluster;

                // Accumulate results for this thread
                local_nPoints[bestCluster]++;
                local_sumX[bestCluster] += point.x;
                local_sumY[bestCluster] += point.y;
            }

            // Combine local results into global
            #pragma omp critical
            {
                for (int i = 0; i < k; ++i) {
                    nPoints[i] += local_nPoints[i];
                    sumX[i] += local_sumX[i];
                    sumY[i] += local_sumY[i];
                }
            }
        }

        // Update centroids
        bool converged = true;

        for (int i = 0; i < k; ++i) {
            if (nPoints[i] > 0) {
                double newX = sumX[i] / nPoints[i];
                double newY = sumY[i] / nPoints[i];

                if (std::abs(newX - centroids[i].x) > tolerance || std::abs(newY - centroids[i].y) > tolerance) {
                    converged = false;
                }

                centroids[i].x = newX;
                centroids[i].y = newY;
            }
        }

        if (converged) {
            std::cout << "Converged after " << epoch + 1 << " epochs." << std::endl;
            break;
        }
    }
}

std::vector<Point> readCSV() {
    std::cout << "Reading CSV file..." << std::endl;
    std::vector<Point> points;
    std::string line;
    std::ifstream file("resources/income_score.csv");
    std::cout << (file.is_open() ? "File Opened" : "Could not open the file!") << std::endl;
    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string bit;
        double x, y;

        // Read the first value (x)
        std::getline(lineStream, bit, ',');
        x = std::stod(bit);

        // Read the second value (y)
        std::getline(lineStream, bit, ',');
        y = std::stod(bit);

        points.emplace_back(x, y);
    }

    return points;
}

void draw_chart_gnu(std::vector<Point> &points){

    std::ofstream outfile("data.txt");

    for(int i = 0; i < points.size(); i++){

        Point point = points[i];
        outfile << point.x << " " << point.y << " " << point.cluster << std::endl;

    }

    outfile.close();
    system("gnuplot -p -e \"plot 'data.txt' using 1:2:3 with points pt 7 palette notitle\"");
    remove("data.txt");

}

int main() {
    /*std::vector<Point> points = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0}, {6.0, 8.0}, {7.0, 9.0}, {8.0, 8.0}
    };*/
    std::vector<Point> points = readCSV();
    draw_chart_gnu(points);
    /*std::vector<double> x, y;
    for (const auto& point : points) {
        x.push_back(point.x);
        y.push_back(point.y);
    }
    matplot::scatter(x,y)->marker_face(true);
    matplot::xlabel("Annual Income");
    matplot::ylabel("Spending Score (1-100)");
    matplot::title("Customer Segmentation");
    matplot::show();*/

    int k = 5;
    int epochs = 10000;
    double tolerance = 0.00000000001;
    double runtime = omp_get_wtime();
    kmeans(points, k, epochs, tolerance);
    //parallel_kmeans(points, k, epochs, tolerance);
    runtime = omp_get_wtime() - runtime;
    for (const auto& point : points) {
        std::cout << "Point (" << point.x << ", " << point.y << ") is in cluster " << point.cluster << std::endl;
    }
    printf("\n runtime: %f  \n", runtime);
    draw_chart_gnu(points);

    // Plot points with different colors based on cluster
    /*x.clear();
    /y.clear();
    std::vector<int> colors;
    for (const auto& point : points) {
        x.push_back(point.x);
        y.push_back(point.y);
        colors.push_back(point.cluster);
    }
    matplot::scatter(x, y, std::vector<double>{}, colors)->marker_face(true);
    matplot::show();*/

    return 0;
}