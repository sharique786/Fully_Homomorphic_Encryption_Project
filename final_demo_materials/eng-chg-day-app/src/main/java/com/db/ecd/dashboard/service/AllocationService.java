package com.db.ecd.dashboard.service;

import com.db.ecd.dashboard.constant.IdeaStatus;
import com.db.ecd.dashboard.constant.UserRole;
import com.db.ecd.dashboard.constant.UserSet;
import com.db.ecd.dashboard.entity.Idea;
import com.db.ecd.dashboard.entity.TeamAllocation;
import com.db.ecd.dashboard.entity.User;
import com.db.ecd.dashboard.entity.UserPreference;
import com.db.ecd.dashboard.repository.IdeaRepository;
import com.db.ecd.dashboard.repository.TeamAllocationRepository;
import com.db.ecd.dashboard.repository.UserPreferenceRepository;
import com.db.ecd.dashboard.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class AllocationService {

    private final TeamAllocationRepository allocationRepository;
    private final UserRepository userRepository;
    private final IdeaRepository ideaRepository;
    private final UserPreferenceRepository preferenceRepository;

    /**
     * Main allocation algorithm that distributes users to ideas based on preferences
     * Ensures equal distribution within each user set
     */
    @Transactional
    public List<TeamAllocation> allocateUsersToIdeas(UserSet userSet) {
        log.info("Starting allocation for user set: {}", userSet);

        // Get approved ideas for this user set
        List<Idea> ideas = ideaRepository.findByStatusAndUserSet(IdeaStatus.APPROVED, userSet);

        // Get normal users for this user set
        List<User> users = userRepository.findByRoleAndUserSet(UserRole.NORMAL, userSet);

        if (ideas.isEmpty() || users.isEmpty()) {
            log.warn("No ideas or users found for allocation");
            return Collections.emptyList();
        }

        // Calculate target size per idea
        int totalUsers = users.size();
        int totalIdeas = ideas.size();
        int targetSize = totalUsers / totalIdeas;
        int remainder = totalUsers % totalIdeas;

        log.info("Allocating {} users to {} ideas. Target: {}, Remainder: {}",
                totalUsers, totalIdeas, targetSize, remainder);

        // Initialize allocation map
        Map<Long, List<User>> allocations = new HashMap<>();
        for (Idea idea : ideas) {
            allocations.put(idea.getId(), new ArrayList<>());
        }

        // Track allocated users
        Set<Long> allocatedUserIds = new HashSet<>();

        // Phase 1: Allocate based on 1st preferences
        allocateByPreference(users, ideas, allocations, allocatedUserIds, 1);

        // Phase 2: Allocate based on 2nd preferences
        allocateByPreference(users, ideas, allocations, allocatedUserIds, 2);

        // Phase 3: Allocate based on 3rd preferences
        allocateByPreference(users, ideas, allocations, allocatedUserIds, 3);

        // Phase 4: Balance the allocation
        balanceAllocations(users, ideas, allocations, allocatedUserIds, targetSize, remainder);

        // Save allocations to database
        List<TeamAllocation> savedAllocations = new ArrayList<>();
        for (Map.Entry<Long, List<User>> entry : allocations.entrySet()) {
            Idea idea = ideas.stream()
                    .filter(i -> i.getId().equals(entry.getKey()))
                    .findFirst()
                    .orElse(null);

            if (idea != null) {
                for (User user : entry.getValue()) {
                    TeamAllocation allocation = new TeamAllocation();
                    allocation.setIdea(idea);
                    allocation.setUser(user);
                    allocation.setApprovedByApprover(false);
                    savedAllocations.add(allocationRepository.save(allocation));
                }
            }
        }

        log.info("Allocation completed. Total allocations: {}", savedAllocations.size());
        return savedAllocations;
    }

    private void allocateByPreference(List<User> users, List<Idea> ideas,
                                      Map<Long, List<User>> allocations,
                                      Set<Long> allocatedUserIds, int preferenceRank) {

        for (User user : users) {
            if (allocatedUserIds.contains(user.getId())) {
                continue;
            }

            List<UserPreference> preferences = preferenceRepository.findByUser(user);
            Optional<UserPreference> preference = preferences.stream()
                    .filter(p -> p.getPreferenceRank() == preferenceRank)
                    .findFirst();

            if (preference.isPresent()) {
                Long ideaId = preference.get().getIdea().getId();
                if (allocations.containsKey(ideaId)) {
                    allocations.get(ideaId).add(user);
                    allocatedUserIds.add(user.getId());
                }
            }
        }
    }

    private void balanceAllocations(List<User> users, List<Idea> ideas,
                                    Map<Long, List<User>> allocations,
                                    Set<Long> allocatedUserIds,
                                    int targetSize, int remainder) {

        // Allocate remaining users to under-allocated ideas
        List<User> unallocatedUsers = users.stream()
                .filter(u -> !allocatedUserIds.contains(u.getId()))
                .toList();

        // Sort ideas by current allocation size (ascending)
        List<Map.Entry<Long, List<User>>> sortedIdeas = allocations.entrySet().stream()
                .sorted(Comparator.comparingInt(e -> e.getValue().size()))
                .collect(Collectors.toList());

        int ideaIndex = 0;
        for (User user : unallocatedUsers) {
            Map.Entry<Long, List<User>> ideaEntry = sortedIdeas.get(ideaIndex);
            ideaEntry.getValue().add(user);
            allocatedUserIds.add(user.getId());

            ideaIndex = (ideaIndex + 1) % sortedIdeas.size();
        }

        // Rebalance over-allocated ideas
        boolean needsRebalancing = true;
        int maxIterations = 10;
        int iteration = 0;

        while (needsRebalancing && iteration < maxIterations) {
            needsRebalancing = false;
            iteration++;

            sortedIdeas = allocations.entrySet().stream()
                    .sorted(Comparator.comparingInt(e -> e.getValue().size()))
                    .toList();

            for (int i = sortedIdeas.size() - 1; i >= 0; i--) {
                Map.Entry<Long, List<User>> overAllocated = sortedIdeas.get(i);
                int currentSize = overAllocated.getValue().size();
                int allowedSize = targetSize + (i < remainder ? 1 : 0);

                if (currentSize > allowedSize + 1) {
                    // Find under-allocated idea
                    for (Map.Entry<Long, List<User>> underAllocated : sortedIdeas) {
                        if (underAllocated.getValue().size() < targetSize) {
                            // Move user from over to under
                            User userToMove = findBestUserToMove(
                                    overAllocated.getKey(),
                                    underAllocated.getKey(),
                                    overAllocated.getValue()
                            );

                            if (userToMove != null) {
                                overAllocated.getValue().remove(userToMove);
                                underAllocated.getValue().add(userToMove);
                                needsRebalancing = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    private User findBestUserToMove(Long fromIdeaId, Long toIdeaId, List<User> users) {
        // Prefer moving users who have preference for the target idea
        for (User user : users) {
            List<UserPreference> prefs = preferenceRepository.findByUser(user);
            boolean hasPreferenceForTarget = prefs.stream()
                    .anyMatch(p -> p.getIdea().getId().equals(toIdeaId));

            if (hasPreferenceForTarget) {
                return user;
            }
        }

        // Otherwise return any user
        return users.isEmpty() ? null : users.get(0);
    }

    @Cacheable("allocations")
    public List<TeamAllocation> getAllocations() {
        return allocationRepository.findAll();
    }

    @Transactional
    public void approveAllocations() {
        List<TeamAllocation> allocations = allocationRepository.findByApprovedByApprover(false);
        for (TeamAllocation allocation : allocations) {
            allocation.setApprovedByApprover(true);
            allocation.setApprovedAt(LocalDateTime.now());
            allocationRepository.save(allocation);
        }
    }
}
