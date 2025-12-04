package com.db.ecd.dashboard.repository;

import com.db.ecd.dashboard.constant.IdeaStatus;
import com.db.ecd.dashboard.constant.UserSet;
import com.db.ecd.dashboard.entity.Idea;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface IdeaRepository extends JpaRepository<Idea, Long> {
    List<Idea> findByStatus(IdeaStatus status);
    List<Idea> findBySelectedForReview(boolean selected);
    List<Idea> findByApproved(boolean approved);
    List<Idea> findByUserSet(UserSet userSet);
    List<Idea> findByStatusAndUserSet(IdeaStatus status, UserSet userSet);
}
