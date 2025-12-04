package com.db.ecd.dashboard.controller;

import com.db.ecd.dashboard.entity.*;
import com.db.ecd.dashboard.service.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/ideas")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class IdeaController {

    private final IdeaService ideaService;

    @PostMapping
    public ResponseEntity<Idea> submitIdea(@RequestBody Idea idea) {
        return ResponseEntity.ok(ideaService.submitIdea(idea));
    }

    @GetMapping
    public ResponseEntity<List<Idea>> getAllIdeas() {
        return ResponseEntity.ok(ideaService.getAllIdeas());
    }

    @GetMapping("/{id}")
    public ResponseEntity<Idea> getIdea(@PathVariable Long id) {
        return ideaService.getIdeaById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/selected")
    public ResponseEntity<List<Idea>> getSelectedIdeas() {
        return ResponseEntity.ok(ideaService.getSelectedIdeas());
    }

    @GetMapping("/approved")
    public ResponseEntity<List<Idea>> getApprovedIdeas() {
        return ResponseEntity.ok(ideaService.getApprovedIdeas());
    }

    @PostMapping("/select")
    public ResponseEntity<List<Idea>> selectTop10Ideas(@RequestBody List<Long> ideaIds) {
        return ResponseEntity.ok(ideaService.selectTop10Ideas(ideaIds));
    }

    @PostMapping("/{id}/approve")
    public ResponseEntity<Idea> approveIdea(@PathVariable Long id) {
        return ResponseEntity.ok(ideaService.approveIdea(id));
    }

    @PostMapping("/{id}/review")
    public ResponseEntity<Idea> requestReview(@PathVariable Long id, @RequestBody String note) {
        return ResponseEntity.ok(ideaService.requestReview(id, note));
    }

    @PostMapping("/{id}/vote")
    public ResponseEntity<Idea> voteForIdea(@PathVariable Long id) {
        return ResponseEntity.ok(ideaService.voteForIdea(id));
    }

    @GetMapping("/top3")
    public ResponseEntity<List<Idea>> getTop3Ideas() {
        return ResponseEntity.ok(ideaService.getTop3Ideas());
    }
}