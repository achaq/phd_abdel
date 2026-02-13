---
name: fastapi-microservices-serverless
description: Develops Python FastAPI applications following microservices and serverless architecture patterns. Use when working with FastAPI, Python microservices, serverless deployment, API gateways, async workers, distributed systems, or cloud-native applications.
---

# FastAPI Microservices & Serverless Development

## Core Architecture Principles

- Design services to be **stateless**; leverage external storage and caches (Redis) for state persistence
- Implement **API gateways and reverse proxies** (NGINX, Traefik) for handling traffic to microservices
- Use **circuit breakers and retries** for resilient service communication
- Favor **serverless deployment** for reduced infrastructure overhead in scalable environments
- Use **asynchronous workers** (Celery, RQ) for handling background tasks efficiently

## Microservices Architecture

### Service Design

- Design APIs with **clear separation of concerns** to align with microservices principles
- Keep services independent and loosely coupled
- Each service should have a single, well-defined responsibility

### API Gateway Integration

- Integrate FastAPI services with API Gateway solutions (Kong, AWS API Gateway)
- Use API Gateway for:
  - Rate limiting
  - Request transformation
  - Security filtering
  - Request routing

### Inter-Service Communication

- Use **message brokers** (RabbitMQ, Kafka) for event-driven architectures
- Implement pub/sub patterns for decoupled service communication
- Prefer asynchronous messaging over synchronous HTTP calls when possible

```python
# Example: Event-driven service communication
from fastapi import FastAPI
from pydantic import BaseModel
import aiokafka

app = FastAPI()

class OrderEvent(BaseModel):
    order_id: str
    status: str

async def publish_event(event: OrderEvent):
    producer = aiokafka.AIOKafkaProducer(bootstrap_servers='localhost:9092')
    await producer.send('orders', event.model_dump_json().encode())
```

## Serverless & Cloud-Native Patterns

### Optimization for Serverless

- **Minimize cold start times**:
  - Use lightweight dependencies
  - Avoid heavy imports at module level
  - Prefer async/await patterns
  - Keep initialization code minimal

- **Packaging for serverless**:
  - Use lightweight containers (Alpine-based)
  - Consider standalone binaries (PyInstaller, Nuitka)
  - Optimize package size

```python
# Good: Lazy imports to reduce cold start
def get_database():
    from database import Database  # Import only when needed
    return Database()

# Avoid: Heavy imports at module level
from heavy_library import HeavyProcessor  # Increases cold start time
```

### Managed Services

- Use **managed services** for databases (AWS DynamoDB, Azure Cosmos DB) to reduce operational overhead
- Leverage serverless databases that scale automatically
- Implement automatic scaling with serverless functions for variable loads

### Deployment Patterns

- Package FastAPI applications for serverless environments (AWS Lambda, Azure Functions, Google Cloud Functions)
- Use containerization (Docker) for consistent deployments
- Implement health checks and graceful shutdowns

## Security & Middleware

### Custom Middleware

- Implement custom middleware for:
  - Detailed logging
  - Request tracing
  - Performance monitoring
  - Error handling

```python
from fastapi import FastAPI, Request
import time
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response
```

### Distributed Tracing

- Use **OpenTelemetry** or similar libraries for distributed tracing in microservices
- Implement correlation IDs for request tracking across services
- Add tracing spans for critical operations

### Security Best Practices

- **OAuth2** for secure API access
- Implement **rate limiting** to prevent abuse
- Add **DDoS protection** at API Gateway level
- Use security headers:
  - CORS configuration
  - Content Security Policy (CSP)
  - X-Frame-Options
  - X-Content-Type-Options

- Apply **content validation** using tools like OWASP Zap
- Validate all inputs using Pydantic models
- Sanitize user inputs to prevent injection attacks

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/api/data")
@limiter.limit("10/minute")
async def get_data(token: str = Depends(oauth2_scheme)):
    # Validate token and return data
    return {"data": "protected"}
```

## Performance & Scalability

### Async Capabilities

- Leverage FastAPI's **async capabilities** for handling large volumes of simultaneous connections
- Use async database drivers (asyncpg, motor)
- Prefer async HTTP clients (httpx, aiohttp)

```python
from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/aggregate")
async def aggregate_data():
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            client.get("http://service1/api/data"),
            client.get("http://service2/api/data"),
            client.get("http://service3/api/data")
        )
    return {"aggregated": results}
```

### Database Optimization

- Optimize backend services for **high throughput and low latency**
- Use databases optimized for read-heavy workloads (Elasticsearch for search)
- Implement read replicas for scaling read operations
- Use connection pooling effectively

### Caching Strategies

- Use **caching layers** (Redis, Memcached) to reduce load on primary databases
- Implement cache-aside pattern for frequently accessed data
- Set appropriate TTLs for cached data
- Invalidate cache on data updates

```python
from fastapi import FastAPI
import redis
import json

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.get("/cached-data/{key}")
async def get_cached_data(key: str):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    # Fetch from database
    data = await fetch_from_db(key)
    redis_client.setex(key, 3600, json.dumps(data))  # Cache for 1 hour
    return data
```

### Load Balancing & Service Mesh

- Apply **load balancing** for distributing traffic across service instances
- Use **service mesh technologies** (Istio, Linkerd) for:
  - Service-to-service communication
  - Fault tolerance
  - Traffic management
  - Security policies

## Monitoring & Logging

### Monitoring

- Use **Prometheus and Grafana** for monitoring FastAPI applications
- Set up alerts for:
  - High error rates
  - Slow response times
  - Resource exhaustion
  - Service availability

- Expose metrics endpoints (`/metrics`) for Prometheus scraping
- Track key metrics:
  - Request rate
  - Response times (p50, p95, p99)
  - Error rates
  - Active connections

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')

app = FastAPI()

@app.middleware("http")
async def metrics_middleware(request, call_next):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        response = await call_next(request)
    return response

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

### Logging

- Implement **structured logging** for better log analysis
- Use JSON format for logs in production
- Include correlation IDs in log entries
- Integrate with centralized logging systems:
  - ELK Stack (Elasticsearch, Logstash, Kibana)
  - AWS CloudWatch
  - Google Cloud Logging
  - Azure Monitor

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log(self, level, message, **kwargs):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))

logger = StructuredLogger(__name__)
logger.log("info", "Request processed", user_id="123", duration_ms=45)
```

## Development Checklist

Before deploying a FastAPI microservice, verify:

- [ ] Service is stateless (no in-memory state)
- [ ] External storage/cache used for state persistence
- [ ] API Gateway integration configured
- [ ] Circuit breakers implemented for external calls
- [ ] Retry logic with exponential backoff
- [ ] Async workers configured for background tasks
- [ ] Message broker integration for inter-service communication
- [ ] Cold start optimization for serverless deployment
- [ ] Security middleware (OAuth2, rate limiting) implemented
- [ ] Distributed tracing configured
- [ ] Caching strategy implemented
- [ ] Monitoring and alerting set up
- [ ] Structured logging configured
- [ ] Health check endpoints available
- [ ] Error handling and graceful degradation
- [ ] Input validation using Pydantic models
- [ ] Database connection pooling configured
- [ ] Load balancing configured (if applicable)
