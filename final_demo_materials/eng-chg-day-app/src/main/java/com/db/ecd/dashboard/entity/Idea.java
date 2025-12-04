package com.db.ecd.dashboard.entity;


import com.db.ecd.dashboard.constant.IdeaStatus;
import com.db.ecd.dashboard.constant.UserSet;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.time.LocalDateTime;

@Entity
@Table(name = "ideas")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Idea {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String title;

    @Column(length = 2000)
    private String description;

    private String category;

    private String impactArea;

    private String estimatedBenefit;

    @Column(nullable = false)
    private String author;

    @Enumerated(EnumType.STRING)
    private UserSet userSet;

    @Enumerated(EnumType.STRING)
    private IdeaStatus status = IdeaStatus.PENDING;

    private boolean selectedForReview = false;

    private boolean approved = false;

    private String disagreementNote;

    private Integer votes = 0;

    @Column(updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    private LocalDateTime updatedAt = LocalDateTime.now();

    @PreUpdate
    public void preUpdate() {
        this.updatedAt = LocalDateTime.now();
    }
}