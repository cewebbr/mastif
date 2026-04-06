# agents.md

## Purpose
This document defines guidelines, constraints, and best practices for coding agents contributing to this project. It ensures consistency, maintainability, and high-quality software design across automated and human contributions.

---

## Core Principles

### 1. Readability First
- Prefer clarity over cleverness.
- Use meaningful names for variables, functions, and classes.
- Keep functions small and focused (Single Responsibility Principle).

### 2. Maintainability
- Write code that is easy to modify and extend.
- Avoid duplication (DRY principle).
- Encapsulate change-prone logic.

### 3. Testability
- Design components to be easily testable.
- Prefer dependency injection over hard-coded dependencies.
- Avoid hidden side effects.

### 4. Consistency
- Follow established project conventions.
- Maintain consistent formatting and structure.

---

## Coding Standards for Agents

### General
- Always validate assumptions explicitly.
- Handle edge cases and failure modes.
- Avoid premature optimization.

### Structure
- Organize code into cohesive modules.
- Separate domain logic from infrastructure concerns.
- Prefer composition over inheritance.

### Error Handling
- Fail fast with clear error messages.
- Avoid silent failures.
- Use structured error types where possible.

---

## Design Guidelines

### GRASP Principles
Agents should apply GRASP (General Responsibility Assignment Software Patterns) where applicable:

- **Information Expert**: Assign responsibilities to the class with the most relevant data.
- **Creator**: A class that aggregates or closely uses another should create it.
- **Controller**: Use a controller to handle system events and coordinate actions.
- **Low Coupling**: Minimize dependencies between components.
- **High Cohesion**: Keep related behavior together.
- **Polymorphism**: Use polymorphism instead of conditionals for behavior variation.
- **Indirection**: Introduce intermediaries to decouple components.
- **Protected Variations**: Isolate areas of likely change behind stable interfaces.

---

### GoF Design Patterns
Agents should recognize opportunities to apply Gang of Four patterns:

#### Creational Patterns
- **Factory Method / Abstract Factory**: For flexible object creation.
- **Builder**: For constructing complex objects step-by-step.
- **Singleton**: Use sparingly; prefer dependency injection.

#### Structural Patterns
- **Adapter**: Integrate incompatible interfaces.
- **Facade**: Simplify complex subsystems.
- **Decorator**: Extend behavior dynamically.
- **Composite**: Represent part-whole hierarchies.

#### Behavioral Patterns
- **Strategy**: Encapsulate interchangeable algorithms.
- **Observer**: React to state changes.
- **Command**: Encapsulate requests as objects.
- **State**: Manage state-dependent behavior.
- **Template Method**: Define algorithm skeletons.

---

## Architectural Patterns (Fowler-Inspired)

Agents should align with established architectural patterns described by Martin Fowler:

### Layered Architecture
- Separate concerns into layers (e.g., presentation, application, domain, infrastructure).
- Ensure clear boundaries and minimal leakage across layers.

### Domain Model
- Encapsulate business logic within domain objects.
- Avoid anemic domain models.

### Service Layer
- Define application boundaries and orchestration logic.

### Repository Pattern
- Abstract data access logic.
- Keep persistence concerns out of domain logic.

### Unit of Work
- Manage transactional consistency across operations.

### Event-Driven Architecture
- Use events to decouple components.
- Prefer asynchronous communication where appropriate.

### Microservices (when applicable)
- Design services around business capabilities.
- Ensure clear contracts and bounded contexts.

---

## Decision Heuristics for Agents

When generating or modifying code, agents should:

1. **Assess Complexity**
   - Simple problem → simple solution.
   - Complex domain → structured patterns.

2. **Choose Patterns Judiciously**
   - Do not over-engineer.
   - Apply patterns only when they reduce complexity or improve clarity.

3. **Preserve Existing Design Integrity**
   - Respect current architecture unless there is a strong reason to refactor.

4. **Refactor Incrementally**
   - Improve code in small, safe steps.

---

## Anti-Patterns to Avoid

- God objects
- Spaghetti code
- Overuse of singletons
- Deep inheritance hierarchies
- Tight coupling between modules
- Premature abstraction

---

## Documentation Expectations

- Document intent, not implementation.
- Provide context for non-obvious decisions.
- Keep documentation up to date with code changes.

---

## Testing Guidelines

- Write unit tests for core logic.
- Use integration tests for system behavior.
- Mock external dependencies appropriately.

---

## Security & Reliability

- Validate all inputs.
- Avoid exposing sensitive data.
- Handle failures gracefully.

---

## Final Notes for Agents

- Prioritize clarity, correctness, and maintainability.
- Prefer well-known patterns over ad-hoc solutions.
- When uncertain, choose the simplest design that satisfies requirements.



---

## Python-Specific Guidelines for Agents

### Language & Style
- Follow **PEP 8** for formatting and naming conventions.
- Use **type hints** (PEP 484) consistently; prefer static analysis compatibility (mypy/pyright).
- Use **dataclasses** (`@dataclass`) for simple data carriers.
- Prefer f-strings over older formatting styles.

### Project Structure
- Use a clear package layout:
  ```
  project/
    domain/
    application/
    infrastructure/
    interfaces/ (or api/)
    tests/
  ```
- Avoid circular imports by respecting layer boundaries.

### Dependency Management
- Use dependency injection via constructor parameters or lightweight containers.
- Avoid global state and module-level singletons.

### Error Handling
- Use custom exception classes for domain-specific errors.
- Avoid catching broad exceptions (`except Exception`) unless rethrowing with context.
- Prefer explicit error handling over implicit failure.

### Logging
- Use the `logging` module (not print statements).
- Include structured, contextual information in logs.

---

## Python Design Patterns (Applied)

### Creational

#### Factory (Preferred over direct instantiation when flexible)
```python
class PaymentProcessorFactory:
    def create(self, method: str) -> PaymentProcessor:
        if method == "credit":
            return CreditCardProcessor()
        if method == "paypal":
            return PayPalProcessor()
        raise ValueError("Unsupported method")
```

#### Builder (for complex objects)
```python
@dataclass
class Report:
    title: str
    sections: list[str]

class ReportBuilder:
    def __init__(self):
        self._sections = []

    def add_section(self, section: str):
        self._sections.append(section)
        return self

    def build(self) -> Report:
        return Report(title="Report", sections=self._sections)
```

---

### Structural

#### Adapter
```python
class LegacyService:
    def do_old_thing(self):
        return "old"

class Adapter:
    def __init__(self, service: LegacyService):
        self._service = service

    def execute(self):
        return self._service.do_old_thing()
```

#### Facade
```python
class OrderFacade:
    def __init__(self, payment, inventory, shipping):
        self.payment = payment
        self.inventory = inventory
        self.shipping = shipping

    def place_order(self, order):
        self.inventory.reserve(order)
        self.payment.charge(order)
        self.shipping.ship(order)
```

---

### Behavioral

#### Strategy
```python
class PricingStrategy:
    def calculate(self, amount: float) -> float:
        raise NotImplementedError

class DiscountStrategy(PricingStrategy):
    def calculate(self, amount: float) -> float:
        return amount * 0.9
```

#### Observer (Pythonic via callbacks)
```python
class EventBus:
    def __init__(self):
        self._subscribers = {}

    def subscribe(self, event, handler):
        self._subscribers.setdefault(event, []).append(handler)

    def publish(self, event, data):
        for handler in self._subscribers.get(event, []):
            handler(data)
```

---

## Pythonic Architectural Patterns

### Layered + Domain Model
- Domain layer: pure Python, no framework dependencies.
- Application layer: orchestrates use cases.
- Infrastructure: external systems (DB, APIs).
- Interfaces: FastAPI/CLI/UI.

### Repository Pattern
```python
class UserRepository(Protocol):
    def get(self, user_id: str) -> User: ...

class SqlUserRepository:
    def __init__(self, session):
        self.session = session

    def get(self, user_id: str) -> User:
        return self.session.query(UserModel).get(user_id)
```

### Unit of Work
```python
class UnitOfWork:
    def __enter__(self):
        self.session = create_session()
        return self

    def __exit__(self, *args):
        self.session.close()

    def commit(self):
        self.session.commit()
```

### Dependency Injection (Pythonic)
- Prefer explicit wiring at application boundaries.
- Use frameworks (e.g., FastAPI Depends) only at the edge.

---

## Testing in Python

- Use **pytest** as the default framework.
- Use fixtures for setup/teardown.
- Prefer parametrized tests for coverage.
- Mock external systems using `unittest.mock` or `pytest-mock`.

---

## Tooling Recommendations

- Formatting: `black`
- Linting: `ruff` or `flake8`
- Type checking: `mypy` or `pyright`
- Testing: `pytest`
- Dependency management: `poetry` or `pip-tools`

---

## Python Anti-Patterns to Avoid

- Overusing dynamic features when static clarity is better
- Monkey patching in production code
- Implicit context via globals
- Mixing sync and async code without clear boundaries
- Massive modules instead of cohesive packages

---

## Final Python Notes for Agents

- Prefer explicit over implicit ("The Zen of Python").
- Write idiomatic Python, not Java/C++ in Python syntax.
- Use the standard library before introducing dependencies.
- Keep functions small, composable, and testable.

