#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

double max_range = 100000;
int num_point = 50000;
int num_cluster = 20;
int max_iterations = 20;

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

            #pragma omp for nowait
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

        #pragma omp parallel for reduction(&:converged)
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
    std::ifstream file("resources/age_satisfaction.csv");
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

/**void draw_chart_gnu(std::vector<Point> &points){

    std::ofstream outfile("data.txt");

    for(int i = 0; i < points.size(); i++){

        Point point = points[i];
        outfile << point.x << " " << point.y << " " << point.cluster << std::endl;

    }

    outfile.close();
    system("gnuplot -p -e \"set xlabel 'Annual Income (k$)'; set ylabel 'Spending Score (1-100)'; set palette rgbformulae 22,13,-31; plot 'data.txt' using 1:2:3 with points pt 7 palette notitle\"");
    remove("data.txt");

}**/

void draw_chart_gnu(std::vector<Point> &points, const std::string& filename) {
    std::ofstream outfile("data.txt");
    std::filesystem::create_directory("plots");

    for (int i = 0; i < points.size(); i++) {
        Point point = points[i];
        outfile << point.x << " " << point.y << " " << point.cluster << std::endl;
    }

    outfile.close();

    std::string gnuplot_command = "gnuplot -e \"set terminal png size 800,600; set output 'plots/" + filename + "'; set xlabel 'Annual Income (k$)'; set ylabel 'Spending Score (1-100)'; set palette rgbformulae 22,13,-31; set cbrange [0:20]; plot 'data.txt' using 1:2:3 with points pt 7 palette notitle\"";
    system(gnuplot_command.c_str());

    remove("data.txt");
}


std::vector<Point> init_point(int num_point){

    std::vector<Point> points(num_point);
    Point *ptr = &points[0];

    for(int i = 0; i < num_point; i++){

        Point* point = new Point(rand() % (int)max_range, rand() % (int)max_range);

        ptr[i] = *point;

    }

    return points;

}

// Function to log the performance results of the rendering
void logExecutionDetails(const std::string& filename, const std::vector<std::tuple<int, int, double, double, double>>& results) {
    std::ofstream out(filename);
    out << "points | Threads | Duration (s) | Speedup | Efficiency\n";
    out << "-----------------------------------------------------------\n";
    for (const auto& result : results) {
        out << std::setw(7) << std::get<0>(result) << " | "
            << std::setw(7) << std::get<1>(result) << " | "
            << std::setw(18) << std::fixed << std::setprecision(4) << std::get<2>(result) << " | "
            << std::setw(7) << std::fixed << std::setprecision(2) << std::get<3>(result) << " | "
            << std::setw(9) << std::fixed << std::setprecision(2) << std::get<4>(result) << "\n";
    }
    out.close();
}

int main() {
    std::vector<int> numThreadsList = {1,2,3, 4,5,6,7,8,9,10,11,12,13,14,15,16};
    std::vector<int> numPointsList = {500, 1000, 2000, 3000}; // Lista dei numeri di punti
    std::vector<std::tuple<int, int, double, double, double>> results;

    int k = 20;
    int epochs = 10000;
    double tolerance = 0.00000000001;

    // Leggi i punti dal file CSV una sola volta
    std::vector<Point> points = readCSV();
    numPointsList.push_back(points.size());

    for (int numPoints : numPointsList) {
        // Misura il tempo per kmeans (sequenziale)
        double totalDurationSequential = 0.0;
        int numMeasurements = 5; // Numero di misurazioni per calcolare la durata media
        double avgDurationSequential = 0.0;

        for (int numThreads : numThreadsList) {
            omp_set_num_threads(numThreads);

            double totalDuration = 0.0;

            for (int measurement = 0; measurement < numMeasurements; ++measurement) {
                // Riduci il numero di punti in pointsCopy
                std::vector<Point> pointsCopy(points.begin(), points.begin() + numPoints);
                if(measurement == 0) {
                    draw_chart_gnu(pointsCopy, "initial_points_number_" + std::to_string(numPoints)+ ".png");
                }
                double startParallel = omp_get_wtime();
                if(numThreads== 1) {
                    kmeans(pointsCopy, k, epochs, tolerance);
                } else {
                    parallel_kmeans(pointsCopy, k, epochs, tolerance);
                }
                double endParallel = omp_get_wtime();
                totalDuration += (endParallel - startParallel);
                draw_chart_gnu(pointsCopy, "kmeans_numThreads_" + std::to_string(numThreads)+"_numPoints_"+ std::to_string(numPoints) + "_measurament_n_" + std::to_string(measurement) +".png");
            }

            if(numThreads==1) {
                avgDurationSequential = totalDuration / numMeasurements;
            }

            double avgDurationParallel = totalDuration / numMeasurements;
            double speedup = avgDurationSequential / avgDurationParallel;
            double efficiency = speedup / numThreads;

            std::cout << "Points: " << numPoints << ", Threads: " << numThreads
                      << ", Avg Duration (Parallel): " << avgDurationParallel << "s, Speedup: " << speedup
                      << ", Efficiency: " << efficiency << std::endl;

            results.push_back({numPoints, numThreads, avgDurationParallel, speedup, efficiency});
        }
    }

    logExecutionDetails("performance_log.txt", results);

    return 0;
}
