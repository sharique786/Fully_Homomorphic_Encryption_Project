# Engineering Change Day Dashboard

A comprehensive web application for managing innovation competitions with idea submission, approval workflows, preference management, and intelligent team allocation.

## 🏗️ Architecture

### Technology Stack

**Frontend:**
- React 18 with Hooks
- Tailwind CSS for styling
- Lucide React for icons
- Responsive design

**Backend:**
- Java 17
- Spring Boot 3.2
- Spring Data JPA
- Spring Security
- EhCache for caching
- H2 In-Memory Database (demo)

**Build & Deployment:**
- Maven for build automation
- Docker & Docker Compose
- Embedded React in Spring Boot JAR

## 📋 Features

### User Management
- **3 User Roles**: Normal User, Admin, Approver
- **User Sets**: SET1 and SET2 for geographical distribution
- Secure authentication with BCrypt password hashing

### Idea Submission
- All users can submit ideas with:
    - Title & Description
    - Category (Technology, Process, Innovation, etc.)
    - Impact Area
    - Estimated Benefits
- Real-time idea viewing for all users

### Admin Workflow
1. Review all submitted ideas
2. Select top 10 ideas for approval
3. Distribute ideas across user sets
4. Trigger allocation algorithm
5. Vote for top 3 best ideas

### Approver Workflow
1. Review admin-selected ideas
2. Approve or request review with notes
3. Review and approve team allocations
4. View all ideas (read-only selection)

### User Preference System
- Normal users select up to 3 preferred ideas
- Rank preferences (1st, 2nd, 3rd choice)
- View personal preference history

### Intelligent Allocation Algorithm
**Key Features:**
- Equal distribution of users across ideas
- Respects user preferences (1st, 2nd, 3rd priority)
- Maintains user set segregation (SET1 ↔ SET2)
- Automatic rebalancing for fairness
- Optimized for teams of varying sizes

**Algorithm Steps:**
1. Allocate by 1st preferences
2. Allocate by 2nd preferences
3. Allocate by 3rd preferences
4. Balance remaining users equally
5. Rebalance over/under-allocated teams

### Caching Strategy
- EhCache for fast data retrieval
- Cached entities: Ideas, Users, Preferences, Allocations
- TTL-based expiration (10-30 minutes)
- Cache invalidation on updates

## 🚀 Getting Started

### Prerequisites
- Java 17+
- Maven 3.8+
- Node.js 18+
- Docker (optional)

### Project Structure
```
ecd-dashboard/
├── src/
│   ├── main/
│   │   ├── java/com/ecd/dashboard/
│   │   │   ├── ECDDashboardApplication.java
│   │   │   ├── entity/
│   │   │   │   ├── User.java
│   │   │   │   ├── Idea.java
│   │   │   │   ├── UserPreference.java
│   │   │   │   └── TeamAllocation.java
│   │   │   ├── repository/
│   │   │   ├── service/
│   │   │   │   ├── UserService.java
│   │   │   │   ├── IdeaService.java
│   │   │   │   ├── PreferenceService.java
│   │   │   │   └── AllocationService.java
│   │   │   ├── controller/
│   │   │   └── config/
│   │   ├── resources/
│   │   │   ├── application.properties
│   │   │   ├── ehcache.xml
│   │   │   └── static/ (React build)
│   │   └── frontend/ (React source)
│   │       ├── package.json
│   │       ├── src/
│   │       │   └── App.jsx
│   │       └── public/
│   └── test/
├── pom.xml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Local Development

#### 1. Run Backend Only
```bash
# Build
mvn clean install

# Run
mvn spring-boot:run

# Access
# Backend: http://localhost:8080
# H2 Console: http://localhost:8080/h2-console
```

#### 2. Run with React Development Server
```bash
# Terminal 1 - Backend
mvn spring-boot:run

# Terminal 2 - Frontend
cd src/main/frontend
npm install
npm start

# Access
# React Dev: http://localhost:3000
# Backend: http://localhost:8080
```

#### 3. Build Complete Application
```bash
# This builds React and embeds in Spring Boot JAR
mvn clean package

# Run
java -jar target/dashboard-1.0.0.jar

# Access
# Application: http://localhost:8080
```

### Docker Deployment

#### Build and Run
```bash
# Build Docker image
docker build -t ecd-dashboard:latest .

# Run container
docker run -p 8080:8080 ecd-dashboard:latest

# Or use docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

#### Docker Hub Deployment
```bash
# Tag image
docker tag ecd-dashboard:latest yourusername/ecd-dashboard:latest

# Push to Docker Hub
docker push yourusername/ecd-dashboard:latest

# Pull and run on any machine
docker pull yourusername/ecd-dashboard:latest
docker run -p 8080:8080 yourusername/ecd-dashboard:latest
```

## 🔐 Demo Credentials

| Username | Password | Role | User Set |
|----------|----------|------|----------|
| admin | admin123 | ADMIN | SET1 |
| approver | approver123 | APPROVER | SET1 |
| user_set1_1 to user_set1_45 | user123 | NORMAL | SET1 |
| user_set2_1 to user_set2_35 | user123 | NORMAL | SET2 |

## 📡 API Endpoints

### Authentication
- `POST /api/users/register` - Register new user
- `POST /api/users/login` - Login user
- `GET /api/users` - Get all users

### Ideas
- `POST /api/ideas` - Submit idea
- `GET /api/ideas` - Get all ideas
- `GET /api/ideas/{id}` - Get idea by ID
- `GET /api/ideas/selected` - Get selected ideas
- `GET /api/ideas/approved` - Get approved ideas
- `POST /api/ideas/select` - Select top 10 (Admin)
- `POST /api/ideas/{id}/approve` - Approve idea (Approver)
- `POST /api/ideas/{id}/review` - Request review (Approver)
- `POST /api/ideas/{id}/vote` - Vote for idea (Admin)
- `GET /api/ideas/top3` - Get top 3 ideas

### Preferences
- `POST /api/preferences` - Submit preferences
- `GET /api/preferences/user/{userId}` - Get user preferences

### Allocations
- `POST /api/allocations/generate/{userSet}` - Generate allocations
- `GET /api/allocations` - Get all allocations
- `POST /api/allocations/approve` - Approve allocations

## 🎨 UI Features

- **Responsive Design**: Mobile, tablet, and desktop support
- **Modern UI**: Clean interface with Tailwind CSS
- **Company Branding**: Logo placeholder in header
- **Role-Based Views**: Different interfaces for each role
- **Real-time Updates**: Dynamic data refresh
- **Form Validation**: Client and server-side validation

## 🔧 Configuration

### Database Configuration (H2 → PostgreSQL)

To switch from H2 to PostgreSQL:

1. Update `pom.xml`:
```xml
<dependency>
    <groupId>org.postgresql</groupId>
    <artifactId>postgresql</artifactId>
    <scope>runtime</scope>
</dependency>
```

2. Update `application.properties`:
```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/ecddb
spring.datasource.username=ecd_user
spring.datasource.password=ecd_password
spring.jpa.database-platform=org.hibernate.dialect.PostgreSQLDialect
```

3. Uncomment PostgreSQL service in `docker-compose.yml`

### Cache Configuration

Modify `ehcache.xml` to adjust:
- TTL (Time To Live)
- Heap size
- Cache aliases

### Security Configuration

For production:
1. Enable HTTPS
2. Configure JWT tokens
3. Implement proper authentication
4. Update CORS policies
5. Add rate limiting

## 📊 Allocation Algorithm Details

### Input
- **Ideas**: 5-10 approved ideas per user set
- **Users**: 35-45 normal users per set
- **Preferences**: Each user's top 3 choices

### Output
- Balanced team assignments
- ~9 users per idea (for 45 users, 5 ideas)
- Preference satisfaction maximized

### Example
```
Input: SET1 with 45 users, 5 ideas
Target: 9 users per idea

Idea 1: 10 users (10 submitted 1st preference)
Idea 2: 6 users (6 submitted 1st preference)
Idea 3: 15 users (15 submitted 1st preference)
Idea 4: 9 users (9 submitted 1st preference)
Idea 5: 5 users (5 submitted 1st preference)

After rebalancing:
Idea 1: 9 users (moved 1 to Idea 2 based on 2nd pref)
Idea 2: 9 users (received 3 users from others)
Idea 3: 9 users (moved 6 to others based on their prefs)
Idea 4: 9 users (no change)
Idea 5: 9 users (received 4 users from Idea 3)
```

## 🧪 Testing

```bash
# Run all tests
mvn test

# Run with coverage
mvn test jacoco:report

# Integration tests
mvn verify
```

## 📦 Production Deployment

### AWS EC2
```bash
# SSH into EC2
ssh -i key.pem ec2-user@your-instance

# Install Docker
sudo yum install docker -y
sudo service docker start

# Pull and run
docker pull yourusername/ecd-dashboard:latest
docker run -d -p 80:8080 --name ecd ecd-dashboard:latest
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecd-dashboard
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ecd-dashboard
  template:
    metadata:
      labels:
        app: ecd-dashboard
    spec:
      containers:
      - name: ecd-dashboard
        image: yourusername/ecd-dashboard:latest
        ports:
        - containerPort: 8080
```

## 🐛 Troubleshooting

### Common Issues

1. **Port 8080 already in use**
   ```bash
   # Change port in application.properties
   server.port=8081
   ```

2. **React build fails**
   ```bash
   cd src/main/frontend
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

3. **Database connection errors**
    - Check H2 console at `/h2-console`
    - Verify JDBC URL: `jdbc:h2:mem:ecddb`

4. **Cache issues**
   ```bash
   # Clear cache
   mvn clean
   # Restart application
   ```

## 📄 License

This project is licensed under the MIT License.

## 👥 Support

For issues or questions:
- Create an issue on GitHub
- Contact: support@ecd-dashboard.com

## 🚧 Roadmap

- [ ] Email notifications
- [ ] Export reports (PDF/Excel)
- [ ] Advanced analytics dashboard
- [ ] Mobile app (React Native)
- [ ] Integration with Slack/Teams
- [ ] AI-powered idea recommendations
- [ ] Multi-language support

---

**Built with ❤️ using Spring Boot & React**