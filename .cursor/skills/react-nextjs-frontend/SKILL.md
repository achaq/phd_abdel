---
name: react-nextjs-frontend
description: Develops React, Next.js, TypeScript, and TailwindCSS applications following best practices. Use when working with React, Next.js, JavaScript, TypeScript, HTML, CSS, TailwindCSS, Shadcn, Radix, or modern front-end development.
---

# React & Next.js Front-End Development

## Core Principles

- Follow user requirements carefully and completely
- Think step-by-step: describe the plan in pseudocode before coding
- Write correct, best practice, DRY, bug-free, fully functional code
- Prioritize readability over performance
- Fully implement all requested functionality
- Leave NO todos, placeholders, or missing pieces
- Verify code is complete and finalized
- Include all required imports
- Use proper naming for key components
- Be concise and minimize prose
- If uncertain, say so instead of guessing

## Supported Technologies

- ReactJS
- NextJS
- JavaScript
- TypeScript
- TailwindCSS
- HTML
- CSS
- Shadcn
- Radix UI

## Code Implementation Guidelines

### Control Flow

- **Use early returns** whenever possible to improve readability
- Avoid deeply nested conditionals; prefer guard clauses

```typescript
// Good
const handleSubmit = (data: FormData) => {
  if (!data.email) return;
  if (!data.password) return;
  // Process form
};

// Avoid
const handleSubmit = (data: FormData) => {
  if (data.email) {
    if (data.password) {
      // Process form
    }
  }
};
```

### Styling

- **Always use Tailwind classes** for styling HTML elements
- Avoid inline CSS or `<style>` tags
- Use `class:` instead of ternary operators in class attributes when possible

```typescript
// Good
<div className="flex items-center justify-between p-4">
  <button className={isActive ? "bg-blue-500" : "bg-gray-500"}>
    Click me
  </button>
</div>

// Better (when possible)
<div className="flex items-center justify-between p-4">
  <button className={class: { "bg-blue-500": isActive, "bg-gray-500": !isActive }}>
    Click me
  </button>
</div>
```

### Naming Conventions

- Use **descriptive variable and function names**
- Event handlers must use **"handle" prefix**:
  - `handleClick` for `onClick`
  - `handleKeyDown` for `onKeyDown`
  - `handleSubmit` for `onSubmit`
  - `handleChange` for `onChange`

```typescript
// Good
const handleButtonClick = () => {
  // Handle click
};

const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  // Handle change
};

// Avoid
const click = () => {};
const onChange = () => {};
```

### Function Declarations

- **Use `const` instead of `function`** declarations
- Define types when possible

```typescript
// Good
const toggleMenu = (): void => {
  setIsOpen(!isOpen);
};

const calculateTotal = (items: Item[]): number => {
  return items.reduce((sum, item) => sum + item.price, 0);
};

// Avoid
function toggleMenu() {
  setIsOpen(!isOpen);
}
```

### Accessibility

- Implement **accessibility features** on interactive elements
- Include `tabIndex`, `aria-label`, `onClick`, and `onKeyDown` where appropriate

```typescript
// Good
<button
  tabIndex={0}
  aria-label="Close dialog"
  onClick={handleClose}
  onKeyDown={handleKeyDown}
  className="p-2 rounded hover:bg-gray-100"
>
  ×
</button>

const handleKeyDown = (e: React.KeyboardEvent) => {
  if (e.key === "Enter" || e.key === " ") {
    handleClose();
  }
};
```

### TypeScript

- Define types for function parameters and return values
- Use interfaces for component props
- Prefer type inference when types are obvious

```typescript
// Good
interface ButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
}

const Button: React.FC<ButtonProps> = ({ label, onClick, disabled = false }) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="px-4 py-2 bg-blue-500 text-white rounded"
    >
      {label}
    </button>
  );
};
```

## Development Workflow

1. **Plan first**: Describe the approach in pseudocode or step-by-step plan
2. **Confirm**: Get user confirmation before implementing
3. **Implement**: Write complete, functional code following all guidelines
4. **Verify**: Ensure no todos, placeholders, or missing pieces remain

## Code Quality Checklist

Before finalizing code, verify:

- [ ] All functionality is fully implemented
- [ ] No todos or placeholders remain
- [ ] All imports are included
- [ ] Early returns used where appropriate
- [ ] Tailwind classes used for all styling
- [ ] Event handlers use "handle" prefix
- [ ] Functions use `const` declarations
- [ ] Accessibility attributes included on interactive elements
- [ ] TypeScript types defined where applicable
- [ ] Code is readable and follows DRY principles
- [ ] All components are properly named
