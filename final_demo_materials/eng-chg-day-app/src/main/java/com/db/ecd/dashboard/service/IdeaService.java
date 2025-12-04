package com.db.ecd.dashboard.service;

import com.db.ecd.dashboard.constant.IdeaStatus;
import com.db.ecd.dashboard.entity.Idea;
import com.db.ecd.dashboard.repository.IdeaRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class IdeaService {

    private final IdeaRepository ideaRepository;

    @Transactional
    @CacheEvict(value = "ideas", allEntries = true)
    public Idea submitIdea(Idea idea) {
        idea.setStatus(IdeaStatus.PENDING);
        return ideaRepository.save(idea);
    }

    @Cacheable("ideas")
    public List<Idea> getAllIdeas() {
        return ideaRepository.findAll();
    }

    @Cacheable(value = "idea", key = "#id")
    public Optional<Idea> getIdeaById(Long id) {
        return ideaRepository.findById(id);
    }

    public List<Idea> getIdeasByStatus(IdeaStatus status) {
        return ideaRepository.findByStatus(status);
    }

    public List<Idea> getSelectedIdeas() {
        return ideaRepository.findBySelectedForReview(true);
    }

    public List<Idea> getApprovedIdeas() {
        return ideaRepository.findByApproved(true);
    }

    @Transactional
    @CacheEvict(value = {"ideas", "idea"}, allEntries = true)
    public List<Idea> selectTop10Ideas(List<Long> ideaIds) {
        List<Idea> ideas = ideaRepository.findAllById(ideaIds);
        ideas.forEach(idea -> {
            idea.setSelectedForReview(true);
            idea.setStatus(IdeaStatus.SELECTED);
        });
        return ideaRepository.saveAll(ideas);
    }

    @Transactional
    @CacheEvict(value = {"ideas", "idea"}, allEntries = true)
    public Idea approveIdea(Long ideaId) {
        Optional<Idea> ideaOpt = ideaRepository.findById(ideaId);
        if (ideaOpt.isPresent()) {
            Idea idea = ideaOpt.get();
            idea.setApproved(true);
            idea.setStatus(IdeaStatus.APPROVED);
            return ideaRepository.save(idea);
        }
        throw new RuntimeException("Idea not found");
    }

    @Transactional
    @CacheEvict(value = {"ideas", "idea"}, allEntries = true)
    public Idea requestReview(Long ideaId, String disagreementNote) {
        Optional<Idea> ideaOpt = ideaRepository.findById(ideaId);
        if (ideaOpt.isPresent()) {
            Idea idea = ideaOpt.get();
            idea.setStatus(IdeaStatus.REVIEW_REQUESTED);
            idea.setDisagreementNote(disagreementNote);
            return ideaRepository.save(idea);
        }
        throw new RuntimeException("Idea not found");
    }

    @Transactional
    @CacheEvict(value = {"ideas", "idea"}, allEntries = true)
    public Idea voteForIdea(Long ideaId) {
        Optional<Idea> ideaOpt = ideaRepository.findById(ideaId);
        if (ideaOpt.isPresent()) {
            Idea idea = ideaOpt.get();
            idea.setVotes(idea.getVotes() + 1);
            return ideaRepository.save(idea);
        }
        throw new RuntimeException("Idea not found");
    }

    public List<Idea> getTop3Ideas() {
        return ideaRepository.findAll().stream()
                .filter(Idea::isApproved)
                .sorted((a, b) -> b.getVotes().compareTo(a.getVotes()))
                .limit(3)
                .toList();
    }
}
