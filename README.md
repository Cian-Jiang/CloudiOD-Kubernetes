# CloudiOD: Cloud-based Object Detection Service

## Project Overview
This project aims to build a web-based system that we call Cloudiod.  It will allow end-users to send an image to a web service hosted by Docker containers and receive a list of objects detected in their uploaded image.  The project will make use of the YOLO (You Only Look Once) library and OpenCV (Open-Source Computer Vision Library) to perform the required image operations/transformations.  Both YOLO and OpenCV are python-based open-source computer vision and machine learning software libraries.  The web service will be hosted as a container in a Kubernetes cluster.  Kubernetes will be used as the container orchestration system.  The object detection web service is also designed to be a RESTful API that can use Python's Flask library.  We are interested in examining the performance of Cloudiod by varying the rate of requests sent to the system (demand) and the number of existing Pods within the Kubernetes cluster (resources).

## Key objectives

- AI-powered object detection using YOLO v3-tiny
- Integrates the pre-trained model into a Flask web service
- RESTful API built with Flask for image upload and processing
- Containerized application using Docker
- Deployed on a Kubernetes cluster in Oracle Cloud Infrastructure (OCI)
- Performance testing under various load conditions

## Technologies Used

- Python
- Flask
- YOLO v3-tiny (AI object detection model)
- OpenCV
- Docker
- Kubernetes
- Oracle Cloud Infrastructure (OCI)

## Project Structure

- `object_detection.py`: The main web service implementation, including AI model integration
- `Dockerfile`: Instructions for building the Docker image with all necessary AI dependencies
- `my-deployment.yaml`: Kubernetes deployment configuration for scaling the AI service
- `service.yaml`: Kubernetes service configuration for exposing the AI endpoint
- `Cloudiod_client.py`: Client script for testing the AI service
- `char.py`: Script for generating performance charts of the AI service under different loads

## Performance Testing Report

The experimental results are presented in the form of two two-dimensional line charts. One of the charts represents the relationship between the average response time of the service for the local client with different numbers of threads and the number of pods. The other chart shows the corresponding data for the Nectar client.


Local Client Results

![image](https://github.com/user-attachments/assets/8e4bd779-7876-42b6-b804-4ed84566111e)
Figure 1: Relationship between average response time and number of pods (different thread counts, local client)


Nectar Client Results

![image](https://github.com/user-attachments/assets/36d8db30-91b6-450a-9f89-5518d8c67303)
Figure 2: Relationship between average response time and number of pods (different thread counts, Nectar client)


From the plotted charts, we can make the following observations:
1.	With different numbers of threads, as the number of pods increases, the average response time for both the local client and the Nectar client generally decreases. This indicates that, under the experimental conditions, the performance of the distributed system improves with an increase in the number of pods.
2.	If the number of pods is fixed, the response time decreases as the number of threads increases. This suggests that increasing the number of concurrent threads can further improve system performance.
3.	For both clients, the most significant decrease in response time occurs at higher thread counts (e.g., 8 or 16 threads). This may indicate that the performance advantage of the distributed system is more pronounced under high concurrency conditions.
4.	However, it is important to note that when the thread count is high but the number of pods is low, errors occur during execution. In my experimental setup, a single pod can handle up to 4 threads. This reminds us that under high concurrency situations, distributed systems may face challenges related to concurrency control, load balancing, or resource allocation, leading to some request processing failures. We should be particularly cautious in these cases.
5.	The average response time for the Nectar client is generally higher than that of the local client. This is likely due to the Nectar client and server being connected via a public network, resulting in greater network latency.
6.	When the number of threads is 1, the response time for both the local client and the Nectar client fluctuates with the increase in the number of pods. This may suggest that, under low concurrency scenarios, the impact of increasing the number of pods on improving response time is limited. This could be due to the communication and coordination overhead between multiple pods offsetting performance gains.
7.	For some specific thread counts (e.g., 8 and 16 threads), in both the local client and Nectar client, the response time decreases more significantly at higher pod counts (e.g., 16 pods) compared to lower pod counts (e.g., 8 pods). This indicates that when the concurrent thread count is high, the system may be better able to fully utilize distributed computing resources, further improving performance.

Through the analysis above, we can see that increasing the number of pods and client threads in a distributed system has a positive effect on improving system performance. This can help cope with a large number of concurrent requests, thereby enhancing the processing capabilities for compute-intensive tasks such as object detection. However, it is important to note that in practical deployments, there may be a need to balance the relationships among network latency, pod quantity, thread count, and system overhead to find the optimal balance between performance and resource utilization. Additionally, for the runtime errors encountered during the experiment, it is necessary to address potential issues related to concurrency, load balancing, and resource allocation within the system to ensure the stability and reliability of the distributed system.
In conclusion, this report has shed light on the factors influencing the performance of a distributed computing system deployed on a Kubernetes cluster for an object detection web service. The experimental results demonstrate the positive impact of increasing the number of pods and client threads on system performance, while also highlighting the need for careful consideration of network latency, system overhead, and potential issues related to concurrency and resource allocation. By understanding these factors and balancing them effectively, we can optimize their distributed

## Future Improvements

  Implement HTTPS for secure image data transmission
  Develop a standardized API for easier integration of the AI service
  Optimize the system for the latest versions of Docker and Kubernetes
  Explore using more recent versions of YOLO for improved detection accuracy



