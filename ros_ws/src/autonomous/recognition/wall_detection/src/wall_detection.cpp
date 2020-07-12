#include "wall_detection.h"
#include "walls.h"
#include <circle_fit.h>
#include <dynamic_reconfigure/server.h>
#include <wall_detection/wall_detectionConfig.h>

WallDetection::WallDetection()
    : m_private_node_handle("~")
{
    std::string topicClusters;
    std::string topicWalls;
    std::string topicObstacles;

    if (!this->m_private_node_handle.getParamCached("topic_input_clusters", topicClusters))
        topicClusters = TOPIC_VOXEL_;

    if (!this->m_private_node_handle.getParamCached("topic_output_walls", topicWalls))
        topicWalls = TOPIC_WALLS_;

    if (!this->m_private_node_handle.getParamCached("topic_output_obstacles", topicObstacles))
        topicObstacles = topicObstacles;

    this->m_voxel_subscriber =
        m_node_handle.subscribe<pcl::PointCloud<pcl::PointXYZRGBL>>(topicClusters, 1,
                                                                    &WallDetection::wallDetection_callback, this);

    this->m_wall_publisher = m_node_handle.advertise<pcl::PointCloud<pcl::PointXYZRGBL>>(topicWalls, 1);

    this->m_obstacles_publisher = m_node_handle.advertise<pcl::PointCloud<pcl::PointXYZRGBL>>(topicObstacles, 1);

    m_wall_radius = 3;

    m_dyn_cfg_server.setCallback(
        [&](wall_detection::wall_detectionConfig& cfg, uint32_t) { m_wall_radius = cfg.wall_search_radius; });
}

void WallDetection::wallDetection_callback(const pcl::PointCloud<pcl::PointXYZRGBL>::ConstPtr& inputVoxels)
{

    frameID = inputVoxels->header.frame_id;

    std::unordered_map<uint32_t, std::vector<pcl::PointXYZRGBL>*> clustersUsed;
    // map cluster ids in label to map with cluster id as key and pointvector as value
    for (size_t i = 0; i < inputVoxels->points.size(); i++)
    {
        if (clustersUsed.count(inputVoxels->points[i].label) > 0)
        {
            clustersUsed[inputVoxels->points[i].label]->push_back(inputVoxels->points[i]);
        }
        else
        {
            std::vector<pcl::PointXYZRGBL>* tmp = new std::vector<pcl::PointXYZRGBL>();
            tmp->push_back(inputVoxels->points[i]);
            clustersUsed.insert({ inputVoxels->points[i].label, tmp });
        }
    }

    // determine maximum left and right clusters in a radius
    std::pair<int64_t, int64_t> wallIds = determineWallIDs(clustersUsed, m_wall_radius); // these ids are the walls

    std::vector<pcl::PointXYZRGBL>* leftWall = clustersUsed[wallIds.first];
    std::vector<pcl::PointXYZRGBL>* rightWall = clustersUsed[wallIds.second];

    std::vector<uint32_t> ignoreIDs;
    ignoreIDs.push_back(wallIds.first);
    ignoreIDs.push_back(wallIds.second);

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> additional_wall_ids =
        addClustersOnRegression(clustersUsed, ignoreIDs, leftWall, rightWall);

    for (auto id : additional_wall_ids.first)
    {
        leftWall->insert(leftWall->end(), clustersUsed[id]->begin(), clustersUsed[id]->end());
    }

    for (auto id : additional_wall_ids.second)
    {
        rightWall->insert(rightWall->end(), clustersUsed[id]->begin(), clustersUsed[id]->end());
    }

    ignoreIDs.insert(ignoreIDs.end(), additional_wall_ids.first.begin(), additional_wall_ids.first.end());
    ignoreIDs.insert(ignoreIDs.end(), additional_wall_ids.second.begin(), additional_wall_ids.second.end());

    // publish only the clusters with ids equal to the walls, but only if the id is > 0
    if (wallIds.first >= 0 && wallIds.second >= 0)
        publishWall(leftWall, rightWall);

    // publish all other clusters
    publishObstacles(clustersUsed, ignoreIDs);

    // clean up
    for (auto itr = clustersUsed.begin(); itr != clustersUsed.end(); ++itr)
        delete itr->second;
}

Circle WallDetection::fitWall(std::vector<pcl::PointXYZRGBL>* wall)
{
    std::vector<Point> wallPointCloud;
    wallPointCloud.resize(wall->size());
    for (size_t i = 0; i < wall->size(); i++)
    {
        wallPointCloud[i].x = (*wall)[i].x;
        wallPointCloud[i].y = (*wall)[i].y;
    }
    return CircleFit::hyperFit(wallPointCloud);
}

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> WallDetection::addClustersOnRegression(
    std::unordered_map<uint32_t, std::vector<pcl::PointXYZRGBL>*> mapClusters, std::vector<uint32_t> inputIgnoreIDs,
    std::vector<pcl::PointXYZRGBL>* leftWall, std::vector<pcl::PointXYZRGBL>* rightWall)
{
    Circle leftCircle = fitWall(leftWall);
    Circle rightCircle = fitWall(rightWall);

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> additional_wall_ids;

    double distanceThreshold = 0.4;
    uint32_t scoreThreshold = 3;

    for (auto cluster_wall : mapClusters)
    {
        uint32_t clusterID = cluster_wall.first;
        if (std::find(inputIgnoreIDs.begin(), inputIgnoreIDs.end(), clusterID) != inputIgnoreIDs.end())
            continue;

        std::vector<pcl::PointXYZRGBL>* cluster = cluster_wall.second;

        uint32_t leftScore = 0, rightScore = 0;

        for (size_t i = 0; i < cluster->size(); i++)
        {
            Point p = { (*cluster)[i].x, (*cluster)[i].y };

            if (leftCircle.getDistance(p) < distanceThreshold)
                leftScore++;

            if (rightCircle.getDistance(p) < distanceThreshold)
                rightScore++;
        }

        std::cout << "ID: " << clusterID << " Left: " << leftScore << " Right: " << rightScore << std::endl;

        if (leftScore > scoreThreshold || rightScore > scoreThreshold)
        {
            if (leftScore > rightScore)
            {
                additional_wall_ids.first.push_back(clusterID);
            }
            else if (rightScore > leftScore)
            {
                additional_wall_ids.second.push_back(clusterID);
            }
        }
    }

    return additional_wall_ids;
}

void WallDetection::publishObstacles(std::unordered_map<uint32_t, std::vector<pcl::PointXYZRGBL>*> mapClusters,
                                     std::vector<uint32_t> ignoreIDs)
{
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr msg(new pcl::PointCloud<pcl::PointXYZRGBL>);

    msg->header.frame_id = frameID;

    for (auto itr = mapClusters.begin(); itr != mapClusters.end(); ++itr)
    {
        if (std::find(ignoreIDs.begin(), ignoreIDs.end(), itr->first) ==
            ignoreIDs.end()) // ignoreIDs does not contain this id
        {
            for (size_t i = 0; i < itr->second->size(); i++)
            {
                msg->push_back((*itr->second)[i]);
            }
        }
    }
    pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
    m_obstacles_publisher.publish(msg);
}

void WallDetection::publishWall(std::vector<pcl::PointXYZRGBL>* wallLeft, std::vector<pcl::PointXYZRGBL>* wallRight)
{
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr msg(new pcl::PointCloud<pcl::PointXYZRGBL>);
    msg->header.frame_id = frameID;

    for (size_t i = 0; i < wallLeft->size(); i++)
    {
        pcl::PointXYZRGBL tmp;
        tmp.x = (*wallLeft)[i].x;
        tmp.y = (*wallLeft)[i].y;
        tmp.z = (*wallLeft)[i].z;
        tmp.r = (*wallLeft)[i].r;
        tmp.g = (*wallLeft)[i].g;
        tmp.b = (*wallLeft)[i].b;

        tmp.label = WALL_DETECTION_WALL_ID_LEFT;

        msg->push_back(tmp);
    }

    for (size_t i = 0; i < wallRight->size(); i++)
    {
        pcl::PointXYZRGBL tmp;
        tmp.x = (*wallRight)[i].x;
        tmp.y = (*wallRight)[i].y;
        tmp.z = (*wallRight)[i].z;
        tmp.r = (*wallRight)[i].r;
        tmp.g = (*wallRight)[i].g;
        tmp.b = (*wallRight)[i].b;
        tmp.label = WALL_DETECTION_WALL_ID_RIGHT;

        msg->push_back(tmp);
    }

    pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);
    m_wall_publisher.publish(msg);
}

int64_t WallDetection::findLargestCluster(std::unordered_map<uint32_t, std::vector<pcl::PointXYZRGBL>*> clusters,
                                          uint32_t ignoreID)
{
    int64_t largestClusterID = -1;
    uint32_t largestClusterSize = 0;
    for (auto idCluster : clusters)
    {
        if (idCluster.first == ignoreID)
            continue;

        if (idCluster.second->size() > largestClusterSize)
        {
            largestClusterID = idCluster.first;
            largestClusterSize = idCluster.second->size();
        }
    }

    return largestClusterID;
}

std::pair<int64_t, int64_t> WallDetection::determineWallIDs(
    std::unordered_map<uint32_t, std::vector<pcl::PointXYZRGBL>*> mapToCheck, float radius)
{
    float maxLeft = 0;
    float minRight = 0;
    int64_t maxLeftID = -1;
    int64_t minRightID = -1;

    for (auto itr = mapToCheck.begin(); itr != mapToCheck.end(); ++itr)
    {
        for (auto itrVector = itr->second->begin(); itrVector != itr->second->end(); ++itrVector)
        {
            if ((itrVector->y > maxLeft) && (fabsf(itrVector->x) <= radius))
            {
                maxLeft = itrVector->y;
                maxLeftID = itrVector->label;
            }
            if ((itrVector->y < minRight) && (fabsf(itrVector->x) <= radius))
            {
                minRight = itrVector->y;
                minRightID = itrVector->label;
            }
        }
    }

    if (maxLeftID == -1 && minRightID != -1)
    {
        // found a cluster for right but not for left. let's just choose the largest one for the left.
        maxLeftID = findLargestCluster(mapToCheck, minRightID);
    }
    else if (minRightID == -1 && maxLeftID != -1)
    {
        // same but reverse
        minRightID = findLargestCluster(mapToCheck, maxLeftID);
    }

    return std::pair<int64_t, int64_t>(maxLeftID, minRightID);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "wall_detection");
    WallDetection wallDetection;
    ros::spin();
    return EXIT_SUCCESS;
}
