package com.db.ecd.dashboard.config;

import com.db.ecd.dashboard.constant.IdeaStatus;
import com.db.ecd.dashboard.constant.UserRole;
import com.db.ecd.dashboard.constant.UserSet;
import com.db.ecd.dashboard.entity.Idea;
import com.db.ecd.dashboard.entity.User;
import com.db.ecd.dashboard.repository.IdeaRepository;
import com.db.ecd.dashboard.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class DataInitializer implements CommandLineRunner {

    private final UserRepository userRepository;
    private final IdeaRepository ideaRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    public void run(String... args) {
        // Create demo users
        User admin = new User();
        admin.setUsername("admin");
        admin.setPassword(passwordEncoder.encode("admin123"));
        admin.setRole(UserRole.ADMIN);
        admin.setUserSet(UserSet.SET1);
        userRepository.save(admin);

        User approver = new User();
        approver.setUsername("approver");
        approver.setPassword(passwordEncoder.encode("approver123"));
        approver.setRole(UserRole.APPROVER);
        approver.setUserSet(UserSet.SET1);
        userRepository.save(approver);

        // Create normal users for SET1
        for (int i = 1; i <= 45; i++) {
            User user = new User();
            user.setUsername("user_set1_" + i);
            user.setPassword(passwordEncoder.encode("user123"));
            user.setRole(UserRole.NORMAL);
            user.setUserSet(UserSet.SET1);
            userRepository.save(user);
        }

        // Create normal users for SET2
        for (int i = 1; i <= 35; i++) {
            User user = new User();
            user.setUsername("user_set2_" + i);
            user.setPassword(passwordEncoder.encode("user123"));
            user.setRole(UserRole.NORMAL);
            user.setUserSet(UserSet.SET2);
            userRepository.save(user);
        }

        // Create demo ideas
        String[] titles = {
                "AI-Powered Code Review System",
                "Microservices Migration Strategy",
                "Cloud Cost Optimization Tool",
                "Real-time Collaboration Platform",
                "Automated Testing Framework"
        };

        String[] descriptions = {
                "Implement ML-based code review to catch bugs early",
                "Gradual migration from monolith to microservices",
                "Dashboard to monitor and optimize cloud spending",
                "Enable real-time code collaboration for teams",
                "End-to-end automated testing pipeline"
        };

        String[] categories = {"AI/ML", "Architecture", "Cost Reduction", "Collaboration", "Quality"};

        for (int i = 0; i < titles.length; i++) {
            Idea idea = new Idea();
            idea.setTitle(titles[i]);
            idea.setDescription(descriptions[i]);
            idea.setCategory(categories[i]);
            idea.setImpactArea("Efficiency");
            idea.setEstimatedBenefit("20% improvement");
            idea.setAuthor("demo_user");
            idea.setUserSet(i < 3 ? UserSet.SET1 : UserSet.SET2);
            idea.setStatus(IdeaStatus.PENDING);
            ideaRepository.save(idea);
        }

        System.out.println("✅ Database initialized with demo data!");
        System.out.println("👤 Admin: admin / admin123");
        System.out.println("👤 Approver: approver / approver123");
        System.out.println("👤 Users: user_set1_1 to user_set1_45 / user123");
        System.out.println("👤 Users: user_set2_1 to user_set2_35 / user123");
    }
}
