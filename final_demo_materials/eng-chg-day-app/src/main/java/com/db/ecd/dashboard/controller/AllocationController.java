package com.db.ecd.dashboard.controller;

import com.db.ecd.dashboard.constant.UserSet;
import com.db.ecd.dashboard.entity.TeamAllocation;
import com.db.ecd.dashboard.service.AllocationService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/api/allocations")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class AllocationController {

    private final AllocationService allocationService;

    @PostMapping("/generate/{userSet}")
    public ResponseEntity<List<TeamAllocation>> generateAllocations(@PathVariable UserSet userSet) {
        return ResponseEntity.ok(allocationService.allocateUsersToIdeas(userSet));
    }

    @GetMapping
    public ResponseEntity<List<TeamAllocation>> getAllocations() {
        return ResponseEntity.ok(allocationService.getAllocations());
    }

    @PostMapping("/approve")
    public ResponseEntity<Void> approveAllocations() {
        allocationService.approveAllocations();
        return ResponseEntity.ok().build();
    }
}
