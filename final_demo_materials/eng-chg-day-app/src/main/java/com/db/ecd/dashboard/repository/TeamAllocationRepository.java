package com.db.ecd.dashboard.repository;

import com.db.ecd.dashboard.entity.Idea;
import com.db.ecd.dashboard.entity.TeamAllocation;
import com.db.ecd.dashboard.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface TeamAllocationRepository extends JpaRepository<TeamAllocation, Long> {
    List<TeamAllocation> findByIdea(Idea idea);
    List<TeamAllocation> findByUser(User user);
    List<TeamAllocation> findByApprovedByApprover(boolean approved);
    Optional<TeamAllocation> findByUserAndIdea(User user, Idea idea);
    long countByIdea(Idea idea);
}

