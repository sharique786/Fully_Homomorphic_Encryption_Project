package com.db.ecd.dashboard.service;

import com.db.ecd.dashboard.entity.User;
import com.db.ecd.dashboard.entity.UserPreference;
import com.db.ecd.dashboard.repository.UserPreferenceRepository;
import com.db.ecd.dashboard.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;


@Service
@RequiredArgsConstructor
public class PreferenceService {

    private final UserPreferenceRepository preferenceRepository;
    private final UserRepository userRepository;

    @Transactional
    @CacheEvict(value = "preferences", allEntries = true)
    public List<UserPreference> submitPreferences(List<UserPreference> preferences) {
        // Validate max 3 preferences per user
        if (preferences.size() > 3) {
            throw new RuntimeException("Maximum 3 preferences allowed");
        }

        // Delete existing preferences for this user
        if (!preferences.isEmpty()) {
            User user = preferences.get(0).getUser();
            List<UserPreference> existing = preferenceRepository.findByUser(user);
            preferenceRepository.deleteAll(existing);
        }

        return preferenceRepository.saveAll(preferences);
    }

    @Cacheable("preferences")
    public List<UserPreference> getUserPreferences(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return preferenceRepository.findByUser(user);
    }

    public List<UserPreference> getIdeaPreferences(Long ideaId) {
        // This would need Idea entity passed
        return preferenceRepository.findAll(); // Simplified
    }
}
