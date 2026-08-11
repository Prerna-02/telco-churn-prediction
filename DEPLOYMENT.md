# Deployment guide

Four parts, in order:

1. [Push the code to GitHub](#1-github)
2. [Deploy the demo to Render](#2-render) — your portfolio link
3. [Push the image to AWS ECR](#3-aws-ecr)
4. [Run it on AWS ECS Fargate](#4-aws-ecs-fargate)

Run every command from the project root, `e:\Prerna_Gade_SDAIM_Group_Project`.

---

## Before you start

**Delete the old credentials file.** `setup-aws-academy.sh` has an AWS access
key, secret key, and session token written in plain text. `.gitignore` stops it
reaching GitHub, but it should not stay on disk either:

```bash
rm setup-aws-academy.sh
```

Those keys are expired Academy credentials, so nobody can use them now. Just
never put keys in a file that lives next to your code again — step 3 shows the
right way.

---

## 1. GitHub

### 1.1 Check what will be uploaded

`.gitignore` already excludes the things that should not be public:

| Excluded | Why |
|----------|-----|
| `setup-aws-academy.sh` | contains AWS keys |
| `Prerna_Gade_SDAIM_Group_Project/` | a duplicate copy of this whole project |
| `*.docx` | the 2.6 MB report |
| `training.log`, `*.bak`, `__pycache__/` | build noise |

Confirm before committing anything:

```bash
git add -A
git status --short
```

You should see roughly 12 files: the `src/` modules, `Dockerfile`,
`docker-entrypoint.sh`, `requirements*.txt`, `compose.yaml`, `render.yaml`,
`README.md`, `DEPLOYMENT.md`, `models/churn_model.pkl`, the notebook, and the
CSV. **If `setup-aws-academy.sh` appears in that list, stop** and check
`.gitignore` is present.

### 1.2 Commit

```bash
git commit -m "Telco churn prediction: FastAPI + Streamlit, Dockerised"
```

### 1.3 Create the repo and push

You already have the GitHub CLI installed, so this is one command:

```bash
gh auth login
gh repo create telco-churn-prediction --public --source=. --push
```

If you would rather use the website: create an empty repo at
github.com/new (no README, no .gitignore), then:

```bash
git remote add origin https://github.com/<your-username>/telco-churn-prediction.git
git branch -M main
git push -u origin main
```

> The model file is 6 MB, so it pushes normally. No Git LFS needed.

---

## 2. Render

This is the link you put in your portfolio. It is free and it does not depend
on your AWS account staying alive.

1. Go to [dashboard.render.com](https://dashboard.render.com) and sign in with
   GitHub.
2. **New +** → **Blueprint**.
3. Pick your `telco-churn-prediction` repo. Render finds `render.yaml` and
   shows one service, `telco-churn-streamlit`.
4. Click **Apply**. The first build takes about 5–8 minutes (it is installing
   scikit-learn, pandas and Streamlit).
5. When it goes green you get a URL like
   `https://telco-churn-streamlit.onrender.com`.

Put that URL in the README under **Live demo**, commit, and push.

### Things to know about the free plan

- **It sleeps after 15 minutes of inactivity.** The next visitor waits roughly
  50 seconds for it to wake. If someone is reviewing your portfolio, open the
  link yourself a minute beforehand so it is already warm.
- **512 MB RAM — measured, not guessed.** Running the image capped at 512 MB:
  idle Streamlit server 42 MB, 194 MB once the model loads, 200 MB while
  scoring a 2,000-row batch. Roughly 40% of the cap at peak, so memory is not
  a practical constraint here.
- **The first build takes 5–8 minutes** because it compiles nothing but does
  download scikit-learn, pandas and Streamlit.
- Every push to `main` redeploys automatically.

---

## 3. AWS ECR

Now the AWS half, on your own account this time.

### 3.1 Create an IAM user (do not use your root account)

1. AWS Console → **IAM** → **Users** → **Create user**, name it e.g. `deploy-cli`.
2. **Attach policies directly** → tick `AmazonEC2ContainerRegistryFullAccess`
   and `AmazonECS_FullAccess`.
3. Create the user, open it → **Security credentials** → **Create access key**
   → choose **Command Line Interface (CLI)**.
4. Copy the access key ID and secret. This is the only time the secret is shown.

### 3.2 Configure the CLI

```bash
aws configure
```

Paste the key ID and secret when prompted, set region `us-east-1` and output
`json`. This writes them to `~/.aws/credentials`, outside your project folder —
which is why they can never be committed by accident.

Check it worked:

```bash
aws sts get-caller-identity
```

Note the `Account` number it prints; call it `<ACCOUNT_ID>` below.

### 3.3 Create the repository

```bash
aws ecr create-repository --repository-name telco-churn --region us-east-1
```

### 3.4 Log in, build, push

```bash
# 1. Log Docker into your ECR registry (token lasts 12 hours)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# 2. Build for the architecture Fargate runs
docker build --platform linux/amd64 -t telco-churn .

# 3. Tag it with the ECR address
docker tag telco-churn:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/telco-churn:latest

# 4. Push
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/telco-churn:latest
```

The push is about 1.2 GB the first time and a few minutes on a normal
connection. Later pushes only send the layers that changed — if you edit only
`src/`, that is a few hundred KB.

---

## 4. AWS ECS Fargate

> **This costs money.** Fargate is not in the AWS free tier. The smallest task
> (0.25 vCPU, 0.5 GB) running 24/7 is roughly **$9/month**. Adding a load
> balancer would add about $16/month more, so the steps below deliberately skip
> the load balancer and give the task a public IP instead. **Delete the service
> when you are done demoing it** — see step 4.6.

### 4.1 Create a cluster

Console → **ECS** → **Clusters** → **Create cluster**.
Name `churn-cluster`, infrastructure **AWS Fargate (serverless)**, Create.

### 4.2 Create a task definition

**Task definitions** → **Create new task definition**.

- Family: `churn-task`
- Launch type: **AWS Fargate**
- CPU **0.25 vCPU**, Memory **0.5 GB**
- Task execution role: `ecsTaskExecutionRole` (let the console create it if the
  dropdown is empty — it is what lets ECS pull from ECR and write logs)

Container:

- Name: `churn-api`
- Image URI: `<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/telco-churn:latest`
- Port mapping: container port **8000**, protocol TCP
- Environment variables:
  - `APP_MODE` = `api`
  - `PORT` = `8000`

Under **HealthCheck** (optional but nice):

```
CMD-SHELL, curl -f http://localhost:8000/health || exit 1
```

Create.

> Want the Streamlit UI on ECS instead of the API? Same task definition with
> `APP_MODE=ui`. The API is the better story for an ECS deployment, and Render
> already covers the UI.

### 4.3 Create a security group

Console → **EC2** → **Security groups** → **Create security group**.

- Name `churn-sg`, pick your default VPC
- **Inbound rule**: Type *Custom TCP*, Port **8000**, Source **Anywhere-IPv4**
- Leave outbound as-is (it must allow all, so the task can pull the image)

### 4.4 Run it as a service

Back in `churn-cluster` → **Services** → **Create**.

- Compute: **Launch type** → **FARGATE**
- Task definition: `churn-task`, latest revision
- Service name: `churn-service`
- Desired tasks: **1**
- Networking: your default VPC, pick the **public** subnets, security group
  `churn-sg`, and set **Public IP: Turned on** ← without this the task cannot
  pull from ECR and you will get a `CannotPullContainerError`

Create. It takes 2–3 minutes to reach *Running*.

### 4.5 Test it

Open the service → **Tasks** tab → click the running task → copy its
**Public IP**.

```bash
curl http://<PUBLIC_IP>:8000/health
```

Then open `http://<PUBLIC_IP>:8000/docs` in a browser for the interactive API
page, and use **Try it out** on `POST /predict`.

> The public IP changes every time the task restarts. That is fine for a demo.
> A stable URL needs an Application Load Balancer, which is the extra
> $16/month — not worth it here, since Render is your permanent link.

### 4.6 Shut it down when finished

This is the step people forget and then get a surprise bill.

```bash
# Stop paying for compute: scale the service to zero tasks
aws ecs update-service --cluster churn-cluster --service churn-service \
  --desired-count 0 --region us-east-1
```

To remove everything:

```bash
aws ecs delete-service --cluster churn-cluster --service churn-service --force --region us-east-1
aws ecs delete-cluster --cluster churn-cluster --region us-east-1
aws ecr delete-repository --repository-name telco-churn --force --region us-east-1
```

Also set a billing alarm: **Billing** → **Budgets** → create a $5 monthly
budget with an email alert. Do this once and it protects every future project.

---

## Redeploying after a code change

```bash
git add -A && git commit -m "your change" && git push     # Render redeploys itself

# AWS needs the image rebuilt and the service told to pick it up
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker build --platform linux/amd64 -t telco-churn .
docker tag telco-churn:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/telco-churn:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/telco-churn:latest
aws ecs update-service --cluster churn-cluster --service churn-service \
  --force-new-deployment --region us-east-1
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `CannotPullContainerError` | The task has no public IP, or is in a private subnet with no NAT. Turn on **Public IP** in the service networking settings. |
| Task starts then stops immediately | Open the task → **Logs** tab. Usually a missing env var or the image was built for ARM. Rebuild with `--platform linux/amd64`. |
| `exec /app/docker-entrypoint.sh: no such file or directory` | The script got Windows CRLF line endings. `.gitattributes` prevents this; if it happens, run `git add --renormalize . && git commit`. |
| `curl` to the public IP times out | Port 8000 is not open in `churn-sg`, or you used the private IP. |
| Health shows `"status": "degraded"` | `models/churn_model.pkl` is missing from the image. Check it is committed and not caught by `.dockerignore`. |
| Render build fails on memory | Free tier is 512 MB. Check the build log for the actual error first — it is more often a dependency issue than memory. |
| `InvalidVersionWarning` when loading the model | Your scikit-learn is not 1.7.2. `pip install -r requirements.txt`. |
