import os, json, requests
import prod_env


if __name__ == '__main__':
    # Get Version
    versionTag = os.getenv('VERSION_TAG', '')

    # Portainer Creds
    user = os.getenv('PORTAINER_USER', '')
    passw = os.getenv('PORTAINER_PASSW', '')
    stackId = os.getenv('PORTAINER_STACK_ID', '')

    if user == "" or passw == "" or versionTag == "" or stackId == "":
        print("Input parameters are invalid")
        exit(1)

    # Read environment variables
    envs = prod_env.environments

    # Read compose file
    composeFile = f = open("deploy/docker-compose.prod.yml")
    compose = f.read()
    composeFile.close()
    
    # Process env
    procEnv = [
        {"name": "IMAGE_VERSION", "value": versionTag},
    ]
    
    env = prod_env.environments
    for key, value in env.items():
        name = key
        val = value 

        if value[0] == "$":
            val = os.getenv(value[1:], "")

        procEnv.append({"name": name, "value": val})

    # Get JWT token
    data = {
        "password": passw,
         "username":user
    }
    r = requests.post('https://portainer.konkov.xyz/api/auth', data=json.dumps(data))
    jwt = r.json()['jwt']

    data = {
        "env": procEnv,
        "prune": True,
        "stackFileContent": compose
    }
    r = requests.put('https://portainer.konkov.xyz/api/stacks/{}?endpointId=1'.format(stackId), data=json.dumps(data), headers={"Authorization": jwt})
    if r.status_code != 200:
        print("Error", r.status_code, r.text)
        exit(1)
