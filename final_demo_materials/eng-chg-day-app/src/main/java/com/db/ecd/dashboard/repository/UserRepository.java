package com.db.ecd.dashboard.repository;

import com.db.ecd.dashboard.constant.UserRole;
import com.db.ecd.dashboard.constant.UserSet;
import com.db.ecd.dashboard.entity.*;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername(String username);
    List<User> findByRole(UserRole role);
    List<User> findByUserSet(UserSet userSet);
    List<User> findByRoleAndUserSet(UserRole role, UserSet userSet);
}