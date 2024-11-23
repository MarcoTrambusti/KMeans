#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <random>

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

void init_centroids(int k, const std::vector<Point>& points, std::vector<Point>& centroids) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1);

    // Initialize centroids randomly
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points[dis(gen)]);
    }
}

void kmeans(std::vector<Point>& points, std::vector<Point> centroids, int k, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int nPoints[k]= {0};
        double sumX[k] = {0.0};
        double sumY[k] = {0.0};

        #pragma omp parallel default(none) shared(points, centroids, nPoints, sumX, sumY,k) if(omp_get_max_threads() > 1)
        {
            // Assign points to the nearest centroid
            #pragma omp for reduction(+:nPoints,sumX,sumY)
            for (auto& point : points) {
                for (int i = 0; i < k; ++i) {
                    double dist = point.distance(centroids[i]);
                    if (dist < point.minDist) {
                        point.minDist = dist;
                        point.cluster = i;
                    }
                }
                //append data to centroids
                int clusterId = point.cluster;
                nPoints[clusterId]++;
                sumX[clusterId] += point.x;
                sumY[clusterId] += point.y;
                point.minDist = std::numeric_limits<double>::max();  // reset distance
            }

            // Update centroids
            #pragma omp for
            for (int i = 0; i < k; ++i) {
                if (nPoints[i] > 0) {
                    double newX = sumX[i] / nPoints[i];
                    double newY = sumY[i] / nPoints[i];
                    centroids[i].x = newX;
                    centroids[i].y = newY;
                }
            }
        }
    }
}

void draw_chart_gnu(std::vector<Point> &points, const std::string& filename,double minX, double minY, double maxX, double maxY, int numClusters) {
    std::ofstream outfile("data.txt");
    std::filesystem::create_directory("plots");

    for (auto point : points) {
        outfile << point.x *(maxX - minX) + minX<< " " << point.y *(maxY - minY)+minY << " " << point.cluster << std::endl;
    }

    outfile.close();
    std::string gnuplot_command = "gnuplot -e \"set terminal png size 800,600; set output 'plots/" + filename + "'; set xlabel 'Age'; set ylabel 'Total amount spent'; set palette rgbformulae 22,13,-31; set cbrange [0:"+std::to_string(numClusters)+"]; plot 'data.txt' using 1:2:3 with points pt 7 palette notitle\"";
    system(gnuplot_command.c_str());
    remove("data.txt");
}

void logExecutionDetails(const std::string& filename, const std::vector<std::tuple<int, int, double, double, double>>& results) {
    std::ofstream out(filename);
    out << "points,Threads,Duration (s),Speedup,Efficiency\n";
    for (const auto& result : results) {
        out << std::get<0>(result) << ","
            << std::get<1>(result) << ","
            << std::fixed << std::setprecision(4) << std::get<2>(result) << ","
            << std::fixed << std::setprecision(2) << std::get<3>(result) << ","
            << std::fixed << std::setprecision(2) << std::get<4>(result) << "\n";
    }
    out.close();
}

std::vector<Point> readAndNormalizeCSV(const std::string& filename,double& minX, double& minY, double& maxX, double&maxY) {
    std::cout << "Reading CSV file..." << std::endl;
    std::vector<Point> points;
    std::string line;
    std::ifstream file("resources/a.csv");
    std::cout << (file.is_open() ? "File Opened" : "Could not open the file!") << std::endl;
    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string bit;
        double x, y;
        std::getline(lineStream, bit, ',');
        x = std::stod(bit);
        std::getline(lineStream, bit, ',');
        y = std::stof(bit);
        points.emplace_back(x, y);
    }

    // Calcola i valori minimi e massimi per ciascuna dimensione
    minX = std::min_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.x < b.x; })->x;
    maxX = std::max_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.x < b.x; })->x;
    minY = std::min_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.y < b.y; })->y;
    maxY = std::max_element(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.y < b.y; })->y;

    // Normalizza i punti
    for (auto& point : points) {
        point.x = (point.x - minX) / (maxX - minX);
        point.y = (point.y - minY) / (maxY - minY);
    }

    return points;
}

int main() {
    std::vector<int> numThreadsList = {1,2,4,8,16};
    std::vector<int> numPointsList = {500, 7000, 20000, 50000, 126000}; // Lista dei numeri di punti
    std::vector<std::tuple<int, int, double, double, double>> results;
    std::unordered_map<int, int> pointsToClusters = { {500, 5}, {7000, 10}, {20000, 12}, {50000, 15}, {126000, 18}, {301487, 20} };
    std::unordered_map<int, int> pointsToEpochs = { {500, 20}, {7000, 50}, {20000, 70}, {50000, 100}, {126000, 120}, {301487, 200} };

    double minX, minY, maxX, maxY;
    std::vector<Point> points = readAndNormalizeCSV("resources/a.csv", minX, minY, maxX, maxY);
    numPointsList.push_back(points.size());

    for (int numPoints : numPointsList) {
        int numClusters = pointsToClusters[numPoints];
        std::cout << "Number of points: " << numPoints << " -> Number of clusters: " << numClusters << std::endl;
        // Misura il tempo per kmeans (sequenziale)
        double avgDurationSequential = 0.0;
        std::vector<Point> centroids;
        init_centroids(numClusters, points, centroids);

        for (int numThreads : numThreadsList) {
            int numMeasurements = 5;
            omp_set_num_threads(numThreads);
            double totalDuration = 0.0;

            for (int measurement = 0; measurement < numMeasurements; ++measurement) {
                // Riduci il numero di punti in pointsCopy
                std::vector<Point> pointsCopy(points.begin(), points.begin() + numPoints);
                if(measurement == 0) {
                    draw_chart_gnu(pointsCopy, "initial_points_number_" + std::to_string(numPoints)+ ".png",minX, minY, maxX, maxY, numClusters);
                }
                double startParallel = omp_get_wtime();
                kmeans(pointsCopy, centroids,numClusters, pointsToEpochs[numPoints]);
                double endParallel = omp_get_wtime();
                totalDuration += (endParallel - startParallel);

                if(measurement == 0) {
                    draw_chart_gnu(pointsCopy, "kmeans_numThreads_" + std::to_string(numThreads)+"_numPoints_"+ std::to_string(numPoints) + "_measurament_n_" + std::to_string(measurement) +".png", minX, minY, maxX, maxY, numClusters);
                }
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

            results.emplace_back(numPoints, numThreads, avgDurationParallel, speedup, efficiency);
        }
    }

    logExecutionDetails("performance_log.csv", results);

    return 0;
}
