#!/usr/bin/env python3
"""
Test Suite for Cloud Deployment System
=====================================

Comprehensive tests for the cloud and containerized deployment system.
"""

import unittest
import tempfile
import os
import shutil
import json
from unittest.mock import Mock, patch, MagicMock

# Import the system under test
from cloud_deployment_system import (
    CloudDeploymentSystem,
    EnhancedCloudDeploymentSystem,
    DeploymentConfig,
    DeploymentResult,
    CloudPlatform,
    DeploymentType,
    ScalingStrategy,
    DockerContainerBuilder,
    KubernetesDeploymentGenerator,
    CloudPlatformOptimizer,
    DeploymentValidator,
    ServerlessDeploymentGenerator,
    DeploymentMonitor
)

from multi_platform_exporter import ExportFormat


class TestDeploymentConfig(unittest.TestCase):
    """Test deployment configuration"""

    def test_deployment_config_creation(self):
        """Test creating deployment configuration"""
        config = DeploymentConfig(
            platform=CloudPlatform.KUBERNETES,
            deployment_type=DeploymentType.MICROSERVICE,
            scaling_strategy=ScalingStrategy.CPU_BASED,
            min_replicas=2,
            max_replicas=10
        )

        self.assertEqual(config.platform, CloudPlatform.KUBERNETES)
        self.assertEqual(config.deployment_type, DeploymentType.MICROSERVICE)
        self.assertEqual(config.scaling_strategy, ScalingStrategy.CPU_BASED)
        self.assertEqual(config.min_replicas, 2)
        self.assertEqual(config.max_replicas, 10)
        self.assertEqual(config.port, 8080)  # Default value

    def test_deployment_config_defaults(self):
        """Test deployment configuration defaults"""
        config = DeploymentConfig(
            platform=CloudPlatform.DOCKER,
            deployment_type=DeploymentType.REAL_TIME_API,
            scaling_strategy=ScalingStrategy.MEMORY_BASED
        )

        self.assertEqual(config.cpu_request, "500m")
        self.assertEqual(config.memory_request, "1Gi")
        self.assertEqual(config.port, 8080)
        self.assertTrue(config.enable_tls)
        self.assertTrue(config.enable_monitoring)


class TestDockerContainerBuilder(unittest.TestCase):
    """Test Docker container builder"""

    def setUp(self):
        self.config = {"docker_registry": "test-registry.com"}
        self.builder = DockerContainerBuilder(self.config)
        self.deployment_config = DeploymentConfig(
            platform=CloudPlatform.DOCKER,
            deployment_type=DeploymentType.REAL_TIME_API,
            scaling_strategy=ScalingStrategy.CPU_BASED
        )

    def test_generate_dockerfile(self):
        """Test Dockerfile generation"""
        dockerfile = self.builder._generate_dockerfile(
            ExportFormat.TORCHSCRIPT, self.deployment_config
        )

        self.assertIn("FROM python:3.9-slim", dockerfile)
        self.assertIn("EXPOSE 8080", dockerfile)
        self.assertIn("HEALTHCHECK", dockerfile)
        self.assertIn("CMD [\"python\", \"app.py\"]", dockerfile)

    def test_generate_dockerfile_with_gpu(self):
        """Test Dockerfile generation with GPU support"""
        gpu_config = DeploymentConfig(
            platform=CloudPlatform.DOCKER,
            deployment_type=DeploymentType.REAL_TIME_API,
            scaling_strategy=ScalingStrategy.CPU_BASED,
            gpu_required=True
        )

        dockerfile = self.builder._generate_dockerfile(
            ExportFormat.TORCHSCRIPT, gpu_config
        )

        self.assertIn("nvidia/cuda", dockerfile)

    def test_generate_serving_app(self):
        """Test serving application generation"""
        app_code = self.builder._generate_serving_app(
            ExportFormat.TORCHSCRIPT, self.deployment_config
        )

        self.assertIn("class ModelServer", app_code)
        self.assertIn("def predict", app_code)
        self.assertIn("/health", app_code)
        self.assertIn("/metrics", app_code)
        self.assertIn("torch.jit.load", app_code)

    def test_generate_requirements(self):
        """Test requirements.txt generation"""
        requirements = self.builder._generate_requirements(
            ExportFormat.ONNX, self.deployment_config
        )

        self.assertIn("torch>=1.9.0", requirements)
        self.assertIn("fastapi>=0.68.0", requirements)
        self.assertIn("onnxruntime>=1.8.0", requirements)

    @patch('subprocess.run')
    def test_build_container_success(self, mock_run):
        """Test successful container build"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            with open(model_path, 'w') as f:
                f.write("dummy model")

            success, image_tag, error = self.builder.build_container(
                model_path, ExportFormat.TORCHSCRIPT, self.deployment_config
            )

            self.assertTrue(success)
            self.assertIn("nanolm-real_time_api:latest", image_tag)
            self.assertEqual(error, "")

    @patch('subprocess.run')
    def test_build_container_failure(self, mock_run):
        """Test container build failure"""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Build failed"

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            with open(model_path, 'w') as f:
                f.write("dummy model")

            success, image_tag, error = self.builder.build_container(
                model_path, ExportFormat.TORCHSCRIPT, self.deployment_config
            )

            self.assertFalse(success)
            self.assertEqual(image_tag, "")
            self.assertIn("Build failed", error)


class TestKubernetesDeploymentGenerator(unittest.TestCase):
    """Test Kubernetes deployment generator"""

    def setUp(self):
        self.config = {"kubernetes_namespace": "test-namespace"}
        self.generator = KubernetesDeploymentGenerator(self.config)
        self.deployment_config = DeploymentConfig(
            platform=CloudPlatform.KUBERNETES,
            deployment_type=DeploymentType.MICROSERVICE,
            scaling_strategy=ScalingStrategy.CPU_BASED,
            min_replicas=2,
            max_replicas=5
        )

    def test_generate_manifests(self):
        """Test Kubernetes manifest generation"""
        manifests = self.generator.generate_manifests(
            "test-image:latest", self.deployment_config, "test-service"
        )

        self.assertGreater(len(manifests), 0)

        # Check that deployment manifest is generated
        deployment_manifest = manifests[0]
        self.assertIn("kind: Deployment", deployment_manifest)
        self.assertIn("test-image:latest", deployment_manifest)
        self.assertIn("replicas: 2", deployment_manifest)

    def test_generate_deployment_manifest(self):
        """Test deployment manifest generation"""
        deployment_yaml = self.generator._generate_deployment(
            "test-image:latest", self.deployment_config, "test-service"
        )

        self.assertIn("kind: Deployment", deployment_yaml)
        self.assertIn("test-image:latest", deployment_yaml)
        self.assertIn("containerPort: 8080", deployment_yaml)
        self.assertIn("livenessProbe", deployment_yaml)
        self.assertIn("readinessProbe", deployment_yaml)

    def test_generate_service_manifest(self):
        """Test service manifest generation"""
        service_yaml = self.generator._generate_service(
            self.deployment_config, "test-service"
        )

        self.assertIn("kind: Service", service_yaml)
        self.assertIn("port: 80", service_yaml)
        self.assertIn("targetPort: 8080", service_yaml)

    def test_generate_hpa_manifest(self):
        """Test HPA manifest generation"""
        hpa_yaml = self.generator._generate_hpa(
            self.deployment_config, "test-service"
        )

        self.assertIn("kind: HorizontalPodAutoscaler", hpa_yaml)
        self.assertIn("minReplicas: 2", hpa_yaml)
        self.assertIn("maxReplicas: 5", hpa_yaml)
        self.assertIn("averageUtilization: 70", hpa_yaml)


class TestCloudPlatformOptimizer(unittest.TestCase):
    """Test cloud platform optimizer"""

    def setUp(self):
        self.config = {}
        self.optimizer = CloudPlatformOptimizer(self.config)
        self.deployment_config = DeploymentConfig(
            platform=CloudPlatform.AWS,
            deployment_type=DeploymentType.REAL_TIME_API,
            scaling_strategy=ScalingStrategy.CPU_BASED
        )

    def test_optimize_for_aws(self):
        """Test AWS optimizations"""
        optimizations = self.optimizer.optimize_for_platform(
            CloudPlatform.AWS, self.deployment_config
        )

        self.assertIn("instance_types", optimizations)
        self.assertIn("auto_scaling", optimizations)
        self.assertIn("monitoring", optimizations)
        self.assertIn("c5.large", str(optimizations))

    def test_optimize_for_gcp(self):
        """Test GCP optimizations"""
        optimizations = self.optimizer.optimize_for_platform(
            CloudPlatform.GCP, self.deployment_config
        )

        self.assertIn("machine_types", optimizations)
        self.assertIn("auto_scaling", optimizations)
        self.assertIn("monitoring", optimizations)
        self.assertIn("c2-standard-4", str(optimizations))

    def test_optimize_for_serverless(self):
        """Test serverless optimizations"""
        optimizations = self.optimizer.optimize_for_platform(
            CloudPlatform.SERVERLESS, self.deployment_config
        )

        self.assertIn("function_config", optimizations)
        self.assertIn("triggers", optimizations)
        self.assertIn("function_config", optimizations)
        self.assertIn("memory_size", optimizations["function_config"])


class TestServerlessDeploymentGenerator(unittest.TestCase):
    """Test serverless deployment generator"""

    def setUp(self):
        self.config = {}
        self.generator = ServerlessDeploymentGenerator(self.config)
        self.deployment_config = DeploymentConfig(
            platform=CloudPlatform.SERVERLESS,
            deployment_type=DeploymentType.SERVERLESS_FUNCTION,
            scaling_strategy=ScalingStrategy.REQUEST_BASED
        )

    def test_generate_lambda_function(self):
        """Test Lambda function generation"""
        files = self.generator.generate_lambda_function(
            "model.pt", ExportFormat.TORCHSCRIPT, self.deployment_config
        )

        self.assertIn("lambda_function.py", files)
        self.assertIn("template.yaml", files)
        self.assertIn("requirements.txt", files)

    def test_generate_lambda_handler(self):
        """Test Lambda handler generation"""
        handler_code = self.generator._generate_lambda_handler(
            ExportFormat.TORCHSCRIPT, self.deployment_config
        )

        self.assertIn("def lambda_handler", handler_code)
        self.assertIn("torch.jit.load", handler_code)
        self.assertIn("def health_handler", handler_code)

    def test_generate_sam_template(self):
        """Test SAM template generation"""
        template = self.generator._generate_sam_template(self.deployment_config)

        self.assertIn("AWS::Serverless-2016-10-31", template)
        self.assertIn("NanoLMFunction", template)
        self.assertIn("HealthFunction", template)


class TestDeploymentValidator(unittest.TestCase):
    """Test deployment validator"""

    def setUp(self):
        self.config = {}
        self.validator = DeploymentValidator(self.config)
        self.deployment_result = DeploymentResult(
            success=True,
            platform=CloudPlatform.KUBERNETES,
            deployment_type=DeploymentType.MICROSERVICE,
            endpoint_url="http://test-service.example.com"
        )
        self.deployment_config = DeploymentConfig(
            platform=CloudPlatform.KUBERNETES,
            deployment_type=DeploymentType.MICROSERVICE,
            scaling_strategy=ScalingStrategy.CPU_BASED
        )

    @patch('requests.get')
    def test_validate_health_check_success(self, mock_get):
        """Test successful health check validation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response

        health_ok = self.validator._validate_health_check(self.deployment_result)
        self.assertTrue(health_ok)

    @patch('requests.get')
    def test_validate_health_check_failure(self, mock_get):
        """Test health check validation failure"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        health_ok = self.validator._validate_health_check(self.deployment_result)
        self.assertFalse(health_ok)

    @patch('requests.post')
    def test_validate_performance_success(self, mock_post):
        """Test successful performance validation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"inference_time_ms": 100}
        mock_post.return_value = mock_response

        perf_ok = self.validator._validate_performance(
            self.deployment_result, self.deployment_config
        )
        self.assertTrue(perf_ok)


class TestCloudDeploymentSystem(unittest.TestCase):
    """Test main cloud deployment system"""

    def setUp(self):
        self.config = {
            "docker_registry": "test-registry.com",
            "kubernetes_namespace": "test"
        }
        self.system = CloudDeploymentSystem(self.config)
        self.deployment_config = DeploymentConfig(
            platform=CloudPlatform.DOCKER,
            deployment_type=DeploymentType.REAL_TIME_API,
            scaling_strategy=ScalingStrategy.CPU_BASED
        )

    def test_system_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.system.docker_builder)
        self.assertIsNotNone(self.system.k8s_generator)
        self.assertIsNotNone(self.system.cloud_optimizer)
        self.assertIsNotNone(self.system.validator)

    @patch.object(DockerContainerBuilder, 'build_container')
    @patch.object(DeploymentValidator, 'validate_deployment')
    def test_deploy_model_success(self, mock_validate, mock_build):
        """Test successful model deployment"""
        mock_build.return_value = (True, "test-image:latest", "")
        mock_validate.return_value = True

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            with open(model_path, 'w') as f:
                f.write("dummy model")

            result = self.system.deploy_model(
                model_path, ExportFormat.TORCHSCRIPT, self.deployment_config
            )

            self.assertTrue(result.success)
            self.assertEqual(result.platform, CloudPlatform.DOCKER)
            self.assertIn("test-image:latest", result.docker_image)

    def test_create_deployment_package(self):
        """Test deployment package creation"""
        configs = [
            DeploymentConfig(
                platform=CloudPlatform.DOCKER,
                deployment_type=DeploymentType.REAL_TIME_API,
                scaling_strategy=ScalingStrategy.CPU_BASED
            ),
            DeploymentConfig(
                platform=CloudPlatform.KUBERNETES,
                deployment_type=DeploymentType.MICROSERVICE,
                scaling_strategy=ScalingStrategy.REQUEST_BASED
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            with open(model_path, 'w') as f:
                f.write("dummy model")

            output_dir = os.path.join(temp_dir, "package")
            package_info = self.system.create_deployment_package(
                model_path, ExportFormat.TORCHSCRIPT, configs, output_dir
            )

            self.assertIn("platforms", package_info)
            self.assertIn("files", package_info)
            self.assertEqual(len(package_info["platforms"]), 2)
            self.assertGreater(len(package_info["files"]), 0)


class TestEnhancedCloudDeploymentSystem(unittest.TestCase):
    """Test enhanced cloud deployment system"""

    def setUp(self):
        self.config = {"monitoring_enabled": True}
        self.system = EnhancedCloudDeploymentSystem(self.config)
        self.deployment_config = DeploymentConfig(
            platform=CloudPlatform.SERVERLESS,
            deployment_type=DeploymentType.SERVERLESS_FUNCTION,
            scaling_strategy=ScalingStrategy.REQUEST_BASED
        )

    def test_enhanced_system_initialization(self):
        """Test enhanced system initialization"""
        self.assertIsNotNone(self.system.serverless_generator)
        self.assertIsNotNone(self.system.monitor)

    @patch.object(ServerlessDeploymentGenerator, 'generate_lambda_function')
    def test_deploy_serverless(self, mock_generate):
        """Test serverless deployment"""
        mock_generate.return_value = {
            "lambda_function.py": "# Lambda code",
            "template.yaml": "# SAM template",
            "requirements.txt": "torch==1.9.0"
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.pt")
            with open(model_path, 'w') as f:
                f.write("dummy model")

            result = self.system.deploy_serverless(
                model_path, ExportFormat.TORCHSCRIPT, self.deployment_config
            )

            self.assertTrue(result.success)
            self.assertEqual(result.platform, CloudPlatform.SERVERLESS)
            self.assertGreater(len(result.cloud_config_files), 0)


class TestDeploymentMonitor(unittest.TestCase):
    """Test deployment monitor"""

    def setUp(self):
        self.config = {}
        self.monitor = DeploymentMonitor(self.config)
        self.deployment_result = DeploymentResult(
            success=True,
            platform=CloudPlatform.KUBERNETES,
            deployment_type=DeploymentType.MICROSERVICE,
            deployment_id="test-deployment-123"
        )

    def test_collect_metrics(self):
        """Test metrics collection"""
        metrics = self.monitor._collect_metrics(self.deployment_result)

        self.assertIn("timestamp", metrics)
        self.assertIn("cpu_utilization", metrics)
        self.assertIn("memory_utilization", metrics)
        self.assertIn("request_count", metrics)
        self.assertIn("response_time_ms", metrics)

    def test_check_alerts(self):
        """Test alert checking"""
        high_cpu_metrics = {
            "timestamp": 1234567890,
            "cpu_utilization": 85,
            "memory_utilization": 50,
            "error_rate": 2,
            "response_time_ms": 100
        }

        alerts = self.monitor._check_alerts(high_cpu_metrics)

        # Should generate high CPU alert
        cpu_alerts = [a for a in alerts if a["type"] == "high_cpu"]
        self.assertEqual(len(cpu_alerts), 1)
        self.assertEqual(cpu_alerts[0]["severity"], "warning")

    def test_generate_recommendations(self):
        """Test recommendation generation"""
        monitoring_data = {
            "metrics": [
                {"cpu_utilization": 80, "memory_utilization": 90, "response_time_ms": 600},
                {"cpu_utilization": 85, "memory_utilization": 85, "response_time_ms": 700}
            ],
            "alerts": []
        }

        recommendations = self.monitor._generate_recommendations(monitoring_data)

        self.assertGreater(len(recommendations), 0)
        # Should recommend increasing resources due to high utilization
        resource_recommendations = [r for r in recommendations if "increasing" in r.lower()]
        self.assertGreater(len(resource_recommendations), 0)


def run_comprehensive_tests():
    """Run all tests with detailed output"""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDeploymentConfig,
        TestDockerContainerBuilder,
        TestKubernetesDeploymentGenerator,
        TestCloudPlatformOptimizer,
        TestServerlessDeploymentGenerator,
        TestDeploymentValidator,
        TestCloudDeploymentSystem,
        TestEnhancedCloudDeploymentSystem,
        TestDeploymentMonitor
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            error_lines = traceback.split('\n')
            error_msg = error_lines[-2] if len(error_lines) > 1 else str(traceback)
            print(f"  - {test}: {error_msg}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)