# Contributing to DermAI-Melanoma

We welcome contributions to improve the melanoma classification system! This guide will help you get started.

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use GitHub Issues to report bugs
- Include detailed reproduction steps
- Provide system information and error logs
- Check existing issues to avoid duplicates

### 💡 Feature Requests
- Suggest new features or improvements
- Explain the use case and benefits
- Consider backward compatibility
- Discuss implementation approach

### 🔧 Code Contributions
- Bug fixes and improvements
- New training techniques
- UI/UX enhancements
- Documentation updates
- Performance optimizations

### 📚 Documentation
- Tutorial improvements
- Code examples
- Architecture explanations
- Medical context and guidelines

## 🚀 Getting Started

### Development Setup
1. Fork the repository
2. Clone your fork locally
3. Follow the [Installation Guide](INSTALL.md)
4. Create a new branch for your changes

```bash
git checkout -b feature/your-feature-name
```

### Development Workflow
1. Make your changes
2. Test thoroughly
3. Update documentation if needed
4. Commit with clear messages
5. Push to your fork
6. Create a Pull Request

## 📝 Code Guidelines

### Python Code (Training Pipeline)
```python
# Use type hints
def train_model(config: Dict[str, Any]) -> Tuple[float, str]:
    pass

# Follow PEP 8 style guide
# Use descriptive variable names
# Add docstrings to functions

def preprocess_image(image_path: str, target_size: int = 384) -> np.ndarray:
    """
    Preprocess medical image for model input.
    
    Args:
        image_path: Path to input image
        target_size: Target image size in pixels
        
    Returns:
        Preprocessed image array
    """
```

### TypeScript Code (Web Application)
```typescript
// Use proper typing
interface ModelPrediction {
  confidence: number;
  prediction: 'benign' | 'melanoma';
  gradCam?: string;
}

// Follow consistent naming
const handleImageUpload = (file: File): Promise<ModelPrediction> => {
  // Implementation
};

// Add JSDoc comments for complex functions
/**
 * Processes uploaded image and returns melanoma classification
 * @param imageFile - The uploaded image file
 * @returns Promise containing prediction results
 */
```

### Code Style
- **Python**: Use Black formatter and isort for imports
- **TypeScript**: Use Prettier for formatting
- **Comments**: Write clear, concise comments
- **Variable Names**: Use descriptive names, avoid abbreviations

## 🧪 Testing Guidelines

### Python Tests
```bash
cd training
python -m pytest tests/ -v
```

### Web Application Tests
```bash
cd web_app
npm run test
npm run test:coverage
```

### Test Requirements
- Unit tests for new functions
- Integration tests for major features
- Performance benchmarks for model changes
- Browser compatibility tests for UI changes

## 📋 Pull Request Process

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes (or clearly marked)
- [ ] Performance impact considered

### Pull Request Template
```
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other: ___________

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Medical Considerations
- [ ] No impact on medical accuracy
- [ ] Appropriate medical disclaimers maintained
- [ ] Privacy and safety considerations addressed

## Screenshots (if applicable)
[Include before/after screenshots]
```

## 🏥 Medical Ethics Guidelines

### Important Considerations
- **No Clinical Claims**: Don't imply diagnostic capability
- **Privacy First**: Protect patient data at all costs
- **Transparency**: Clearly state limitations
- **Safety**: Include appropriate medical disclaimers

### Required Disclaimers
All medical-related contributions must include:
> ⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. Not for clinical diagnosis.

## 🎨 UI/UX Guidelines

### Design Principles
- **Accessibility**: Follow WCAG 2.1 AA standards
- **Mobile-First**: Design for mobile devices primarily
- **Medical Context**: Professional, clean interface
- **Clear Communication**: Obvious limitations and disclaimers

### Component Guidelines
- Use existing shadcn/ui components when possible
- Maintain consistent spacing and typography
- Include proper loading states
- Add appropriate error handling

## 📊 Performance Guidelines

### Training Performance
- Benchmark changes against baseline
- Consider training time impact
- Test on various hardware configurations
- Document performance improvements/regressions

### Web Application Performance
- Monitor bundle size changes
- Test on various devices and browsers
- Optimize for mobile performance
- Use performance profiling tools

## 🤖 AI/ML Contributions

### Model Improvements
- Validate changes on held-out test set
- Compare against existing benchmarks
- Document hyperparameter changes
- Include ablation studies for major changes

### Data Guidelines
- Respect data licenses and terms
- Maintain patient privacy
- Use appropriate validation splits
- Document data preprocessing changes

## 🛡️ Security Guidelines

### Security Considerations
- No sensitive data in commits
- Validate all user inputs
- Use secure file handling
- Regular dependency updates

### Privacy Protection
- Process data on-device when possible
- No logging of sensitive information
- Clear data retention policies
- GDPR/HIPAA awareness (where applicable)

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Request Reviews**: Code-specific questions

### Response Times
- Issues: We aim to respond within 2-3 days
- Pull Requests: Initial review within 1 week
- Security Issues: Within 24 hours

## 🏆 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes for significant contributions
- Research publications (where applicable)

## 📜 Code of Conduct

We are committed to fostering an inclusive and respectful community. Please:

- Be respectful and professional
- Welcome newcomers and questions
- Focus on constructive feedback
- Respect different viewpoints and experiences
- Report inappropriate behavior

---

Thank you for contributing to DermAI-Melanoma! Together we can improve melanoma detection and save lives. 🎯