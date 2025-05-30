"""
RunPod Deployment Configuration Script
This script helps deploy your IndicF5 TTS API to RunPod
"""

import os
import json
import requests
from typing import Dict, Any

class RunPodDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_template(self, template_config: Dict[str, Any]) -> str:
        """Create a RunPod template for your TTS API"""
        mutation = """
        mutation createTemplate($input: PodTemplateInput!) {
            createTemplate(input: $input) {
                id
                name
            }
        }
        """
        
        variables = {"input": template_config}
        
        response = requests.post(
            self.base_url,
            json={"query": mutation, "variables": variables},
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" not in result:
                template_id = result["data"]["createTemplate"]["id"]
                print(f"✅ Template created successfully! ID: {template_id}")
                return template_id
            else:
                print(f"❌ Error creating template: {result['errors']}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
        
        return None
    
    def deploy_pod(self, template_id: str, pod_config: Dict[str, Any]) -> str:
        """Deploy a pod using the template"""
        mutation = """
        mutation createPod($input: PodCreateInput!) {
            createPod(input: $input) {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
            }
        }
        """
        
        pod_config["templateId"] = template_id
        variables = {"input": pod_config}
        
        response = requests.post(
            self.base_url,
            json={"query": mutation, "variables": variables},
            headers=self.headers
        )
        
        if response.status_code == 200:
            result = response.json()
            if "errors" not in result:
                pod_data = result["data"]["createPod"]
                pod_id = pod_data["id"]
                print(f"✅ Pod deployed successfully! ID: {pod_id}")
                return pod_id
            else:
                print(f"❌ Error deploying pod: {result['errors']}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
        
        return None

def get_template_config(docker_image: str, hf_token: str, api_key: str = None) -> Dict[str, Any]:
    """Generate template configuration for IndicF5 TTS API"""
    
    env_vars = [
        {"key": "HF_TOKEN", "value": hf_token},
        {"key": "PORT", "value": "8000"},
        {"key": "HOST", "value": "0.0.0.0"},
        {"key": "LOG_LEVEL", "value": "INFO"},
        {"key": "MODEL_CACHE_DIR", "value": "/app/model_cache"},
        {"key": "TEMP_DIR", "value": "/app/temp"}
    ]
    
    if api_key:
        env_vars.append({"key": "API_KEY", "value": api_key})
    
    return {
        "name": "IndicF5-TTS-API-Production",
        "imageName": docker_image,
        "readme": """
# IndicF5 TTS API Production Template

This template deploys a production-ready Text-to-Speech API for Indian languages using IndicF5.

## Features:
- Support for 11 Indian languages
- High-quality neural TTS
- RESTful API with FastAPI
- GPU acceleration
- Reference audio cloning

## Endpoints:
- POST /synthesize - Generate speech with uploaded reference audio
- POST /synthesize_with_example - Use preloaded example voices
- POST /synthesize_audio - Get audio file directly
- GET /health - Health check
- GET /examples - Available example voices

## Environment Variables Required:
- HF_TOKEN: Your Hugging Face token (required)
- API_KEY: Optional API authentication key

## Usage:
1. Set your HF_TOKEN in environment variables
2. Deploy the pod
3. Access the API at the provided endpoint
4. Visit /docs for interactive API documentation
        """,
        "dockerArgs": "",
        "containerDiskInGb": 50,
        "volumeInGb": 20,
        "volumeMountPath": "/app/model_cache",
        "ports": "8000/http",
        "env": env_vars,
        "isPublic": False
    }

def get_pod_config(gpu_type: str = "NVIDIA RTX A4000") -> Dict[str, Any]:
    """Generate pod configuration"""
    return {
        "name": f"indicf5-tts-api-{os.urandom(4).hex()}",
        "gpuCount": 1,
        "vcpuCount": 4,
        "memoryInGb": 16,
        "gpuTypeId": gpu_type,
        "supportPublicIp": True,
        "volumeInGb": 20,
        "containerDiskInGb": 50,
        "minVcpuCount": 2,
        "minMemoryInGb": 8
    }

if __name__ == "__main__":
    print("🚀 RunPod IndicF5 TTS API Deployer")
    print("=" * 50)
    
    # Configuration
    RUNPOD_API_KEY = input("Enter your RunPod API key: ").strip()
    HF_TOKEN = input("Enter your Hugging Face token: ").strip()
    DOCKER_IMAGE = input("Enter your Docker image name (e.g., your-username/indicf5-tts:latest): ").strip()
    API_KEY = input("Enter optional API key for authentication (or press Enter to skip): ").strip() or None
    
    if not all([RUNPOD_API_KEY, HF_TOKEN, DOCKER_IMAGE]):
        print("❌ Missing required configuration!")
        exit(1)
    
    # Initialize deployer
    deployer = RunPodDeployer(RUNPOD_API_KEY)
    
    # Create template
    print("\n📝 Creating template...")
    template_config = get_template_config(DOCKER_IMAGE, HF_TOKEN, API_KEY)
    template_id = deployer.create_template(template_config)
    
    if not template_id:
        print("❌ Failed to create template!")
        exit(1)
    
    # Deploy pod
    print("\n🚀 Deploying pod...")
    pod_config = get_pod_config()
    pod_id = deployer.deploy_pod(template_id, pod_config)
    
    if pod_id:
        print(f"\n🎉 Deployment successful!")
        print(f"Pod ID: {pod_id}")
        print(f"Template ID: {template_id}")
        print("\n📋 Next steps:")
        print("1. Wait for the pod to start (check RunPod dashboard)")
        print("2. Get the public IP and port from the dashboard")
        print("3. Access your API at http://PUBLIC_IP:PORT")
        print("4. Visit http://PUBLIC_IP:PORT/docs for API documentation")
    else:
        print("❌ Deployment failed!")
