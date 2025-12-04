package com.db.ecd.dashboard.repository;

import com.db.ecd.dashboard.entity.Idea;
import com.db.ecd.dashboard.entity.User;
import com.db.ecd.dashboard.entity.UserPreference;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface UserPreferenceRepository extends JpaRepository<UserPreference, Long> {
    List<UserPreference> findByUser(User user);
    List<UserPreference> findByIdea(Idea idea);
    List<UserPreference> findByUserAndIdea(User user, Idea idea);
    long countByUser(User user);
}
