"""
Natural language schema compiler for dataset generation.

Compiles natural language dataset specifications into structured generation
templates and validation schemas, acting as a compiler for dataset definitions.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import re


class FieldType(Enum):
    TEXT = "text"
    NUMBER = "number" 
    BOOLEAN = "boolean"
    LIST = "list"
    OBJECT = "object"
    JSON = "json"


class ValidationRule(Enum):
    REQUIRED = "required"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    PATTERN = "pattern"
    CHOICES = "choices"


@dataclass
class FieldSchema:
    """Schema definition for a dataset field."""
    name: str
    type: FieldType
    description: str = ""
    required: bool = True
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    generation_prompt: str = ""
    post_processing: List[str] = field(default_factory=list)


@dataclass
class DatasetSchema:
    """Complete schema for a dataset."""
    name: str
    description: str
    fields: List[FieldSchema]
    global_constraints: Dict[str, Any] = field(default_factory=dict)
    generation_templates: Dict[str, str] = field(default_factory=dict)
    output_format: str = "json"


class SchemaCompiler:
    """Compiles natural language dataset specifications into structured schemas."""
    
    def __init__(self):
        self.built_in_templates = self._load_builtin_templates()
    
    def compile_schema(self, config: Dict) -> DatasetSchema:
        """Main compilation method: convert config to executable schema."""
        
        # Extract basic information
        name = config.get('name', 'Generated Dataset')
        description = config.get('description', '')
        
        # Compile field definitions
        fields = self._compile_fields(config)
        
        # Extract global constraints
        global_constraints = config.get('global_constraints', {})
        
        # Generate templates
        templates = self._compile_templates(config, fields)
        
        return DatasetSchema(
            name=name,
            description=description,
            fields=fields,
            global_constraints=global_constraints,
            generation_templates=templates,
            output_format=config.get('output_format', 'json')
        )
    
    def _compile_fields(self, config: Dict) -> List[FieldSchema]:
        """Compile field definitions from configuration."""
        fields = []
        
        # Check if schema is explicitly defined
        if 'schema' in config:
            return self._parse_explicit_schema(config['schema'])
        
        # Check if using a built-in template
        dataset_type = config.get('dataset_type', 'custom')
        if dataset_type in self.built_in_templates:
            return self._apply_builtin_template(dataset_type, config)
        
        # Parse custom field definitions
        if 'fields' in config:
            return self._parse_custom_fields(config['fields'])
        
        # Fallback to inferred structure
        return self._infer_fields_from_config(config)
    
    def _parse_explicit_schema(self, schema_config: Dict) -> List[FieldSchema]:
        """Parse explicitly defined schema."""
        fields = []
        
        for field_name, field_def in schema_config.items():
            if isinstance(field_def, str):
                # Simple string definition
                field_type, description = self._parse_field_string(field_def)
                field = FieldSchema(
                    name=field_name,
                    type=field_type,
                    description=description
                )
            else:
                # Complex field definition
                field = FieldSchema(
                    name=field_name,
                    type=FieldType(field_def.get('type', 'text')),
                    description=field_def.get('description', ''),
                    required=field_def.get('required', True),
                    validation_rules=field_def.get('validation', {}),
                    examples=field_def.get('examples', []),
                    generation_prompt=field_def.get('prompt', ''),
                    post_processing=field_def.get('post_processing', [])
                )
            
            fields.append(field)
        
        return fields
    
    def _apply_builtin_template(self, dataset_type: str, config: Dict) -> List[FieldSchema]:
        """Apply built-in template for common dataset types."""
        template = self.built_in_templates[dataset_type]
        
        # Apply any customizations from config
        fields = []
        for field_def in template['fields']:
            field = FieldSchema(**field_def)
            
            # Apply config-specific customizations
            if 'field_customizations' in config:
                customizations = config['field_customizations'].get(field.name, {})
                for key, value in customizations.items():
                    setattr(field, key, value)
            
            fields.append(field)
        
        return fields
    
    def _parse_custom_fields(self, fields_config: Dict) -> List[FieldSchema]:
        """Parse custom field definitions."""
        return self._parse_explicit_schema(fields_config)
    
    def _infer_fields_from_config(self, config: Dict) -> List[FieldSchema]:
        """Infer field structure from configuration context."""
        # Default fallback structure
        return [
            FieldSchema(
                name="content",
                type=FieldType.TEXT,
                description="Generated content",
                generation_prompt="Generate content based on the dataset requirements"
            ),
            FieldSchema(
                name="metadata",
                type=FieldType.OBJECT,
                description="Generation metadata",
                required=False
            )
        ]
    
    def _compile_templates(self, config: Dict, fields: List[FieldSchema]) -> Dict[str, str]:
        """Compile generation templates for each field."""
        templates = {}
        
        for field in fields:
            template = self._generate_field_template(field, config)
            templates[field.name] = template
        
        # Add master template
        templates['_master'] = self._generate_master_template(fields, config)
        
        return templates
    
    def _generate_field_template(self, field: FieldSchema, config: Dict) -> str:
        """Generate a specific template for a field."""
        if field.generation_prompt:
            return field.generation_prompt
        
        # Generate template based on field type and context
        base_context = config.get('generation_params', {})
        
        if field.type == FieldType.TEXT:
            return self._generate_text_field_template(field, base_context)
        elif field.type == FieldType.NUMBER:
            return self._generate_number_field_template(field, base_context)
        elif field.type == FieldType.LIST:
            return self._generate_list_field_template(field, base_context)
        else:
            return f"Generate {field.description or field.name}"
    
    def _generate_text_field_template(self, field: FieldSchema, context: Dict) -> str:
        """Generate template for text fields."""
        template_parts = []
        
        # Add field description
        if field.description:
            template_parts.append(f"Generate {field.description}")
        else:
            template_parts.append(f"Generate content for {field.name}")
        
        # Add validation constraints
        if ValidationRule.MIN_LENGTH.value in field.validation_rules:
            min_len = field.validation_rules[ValidationRule.MIN_LENGTH.value]
            template_parts.append(f"Minimum length: {min_len} characters")
        
        if ValidationRule.MAX_LENGTH.value in field.validation_rules:
            max_len = field.validation_rules[ValidationRule.MAX_LENGTH.value]
            template_parts.append(f"Maximum length: {max_len} characters")
        
        # Add examples if available
        if field.examples:
            examples_str = ", ".join(field.examples[:3])
            template_parts.append(f"Examples: {examples_str}")
        
        # Add context from config
        if context:
            relevant_context = self._extract_relevant_context(field.name, context)
            if relevant_context:
                template_parts.append(f"Context: {relevant_context}")
        
        return ". ".join(template_parts) + "."
    
    def _generate_number_field_template(self, field: FieldSchema, context: Dict) -> str:
        """Generate template for number fields."""
        template = f"Generate a numeric value for {field.description or field.name}"
        
        constraints = []
        if ValidationRule.MIN_VALUE.value in field.validation_rules:
            constraints.append(f"minimum {field.validation_rules[ValidationRule.MIN_VALUE.value]}")
        
        if ValidationRule.MAX_VALUE.value in field.validation_rules:
            constraints.append(f"maximum {field.validation_rules[ValidationRule.MAX_VALUE.value]}")
        
        if constraints:
            template += f" ({', '.join(constraints)})"
        
        return template + "."
    
    def _generate_list_field_template(self, field: FieldSchema, context: Dict) -> str:
        """Generate template for list fields."""
        template = f"Generate a list of items for {field.description or field.name}"
        
        if ValidationRule.MIN_LENGTH.value in field.validation_rules:
            min_items = field.validation_rules[ValidationRule.MIN_LENGTH.value]
            template += f" (minimum {min_items} items)"
        
        if ValidationRule.MAX_LENGTH.value in field.validation_rules:
            max_items = field.validation_rules[ValidationRule.MAX_LENGTH.value]
            template += f" (maximum {max_items} items)"
        
        return template + "."
    
    def _generate_master_template(self, fields: List[FieldSchema], config: Dict) -> str:
        """Generate master template that coordinates all fields."""
        template_parts = [
            f"Generate a structured data sample with the following fields:",
            ""
        ]
        
        for field in fields:
            field_desc = field.description or field.name
            template_parts.append(f"- {field.name}: {field_desc}")
        
        template_parts.extend([
            "",
            "Requirements:",
            f"- Output format: {config.get('output_format', 'JSON')}",
            f"- Quality level: {config.get('quality_level', 'standard')}",
        ])
        
        # Add global constraints
        if 'global_constraints' in config:
            constraints = config['global_constraints']
            for constraint, value in constraints.items():
                template_parts.append(f"- {constraint}: {value}")
        
        # Add generation parameters
        if 'generation_params' in config:
            params = config['generation_params']
            template_parts.append("")
            template_parts.append("Generation context:")
            
            for param, value in params.items():
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value[:5])
                    if len(value) > 5:
                        value_str += f" (and {len(value) - 5} more)"
                else:
                    value_str = str(value)
                template_parts.append(f"- {param}: {value_str}")
        
        return "\\n".join(template_parts)
    
    def _extract_relevant_context(self, field_name: str, context: Dict) -> str:
        """Extract context relevant to a specific field."""
        # Simple keyword matching for now
        relevant = []
        
        field_keywords = field_name.lower().split('_')
        
        for key, value in context.items():
            if any(keyword in key.lower() for keyword in field_keywords):
                if isinstance(value, list):
                    relevant.append(f"{key}: {', '.join(str(v) for v in value[:3])}")
                else:
                    relevant.append(f"{key}: {value}")
        
        return "; ".join(relevant) if relevant else ""
    
    def _parse_field_string(self, field_def: str) -> tuple:
        """Parse simple string field definition."""
        # Format: "type: description" or just "description"
        if ':' in field_def:
            type_str, description = field_def.split(':', 1)
            try:
                field_type = FieldType(type_str.strip().lower())
            except ValueError:
                field_type = FieldType.TEXT
            return field_type, description.strip()
        else:
            return FieldType.TEXT, field_def.strip()
    
    def _load_builtin_templates(self) -> Dict:
        """Load built-in templates for common dataset types."""
        return {
            'reasoning': {
                'fields': [
                    {
                        'name': 'problem',
                        'type': 'text',
                        'description': 'The problem or question to solve',
                        'required': True,
                        'validation_rules': {'min_length': 10, 'max_length': 500},
                        'examples': ['What is 15% of 240?', 'Solve for x: 2x + 5 = 17'],
                        'generation_prompt': 'Generate a clear, well-defined problem that requires step-by-step reasoning to solve'
                    },
                    {
                        'name': 'reasoning_steps',
                        'type': 'list',
                        'description': 'Step-by-step reasoning process',
                        'required': True,
                        'validation_rules': {'min_length': 2, 'max_length': 10},
                        'generation_prompt': 'Provide clear, logical steps that lead to the solution. Each step should build on the previous ones.'
                    },
                    {
                        'name': 'answer',
                        'type': 'text',
                        'description': 'The final answer or solution',
                        'required': True,
                        'validation_rules': {'min_length': 1, 'max_length': 200},
                        'generation_prompt': 'Provide the final, correct answer based on the reasoning steps'
                    }
                ]
            },
            'qa_pairs': {
                'fields': [
                    {
                        'name': 'question',
                        'type': 'text',
                        'description': 'The question being asked',
                        'required': True,
                        'validation_rules': {'min_length': 5, 'max_length': 300}
                    },
                    {
                        'name': 'answer',
                        'type': 'text', 
                        'description': 'The answer to the question',
                        'required': True,
                        'validation_rules': {'min_length': 10, 'max_length': 1000}
                    }
                ]
            },
            'instructions': {
                'fields': [
                    {
                        'name': 'instruction',
                        'type': 'text',
                        'description': 'The task instruction or prompt',
                        'required': True,
                        'validation_rules': {'min_length': 10, 'max_length': 500}
                    },
                    {
                        'name': 'response',
                        'type': 'text',
                        'description': 'The expected response or completion',
                        'required': True,
                        'validation_rules': {'min_length': 20, 'max_length': 2000}
                    }
                ]
            },
            'conversations': {
                'fields': [
                    {
                        'name': 'messages',
                        'type': 'list',
                        'description': 'List of conversation messages',
                        'required': True,
                        'validation_rules': {'min_length': 2, 'max_length': 20}
                    }
                ]
            }
        }


class GenerationTemplateEngine:
    """Template engine for generating structured prompts."""
    
    def __init__(self, schema: DatasetSchema):
        self.schema = schema
    
    def generate_prompt(self, field_name: str = None) -> str:
        """Generate a prompt for a specific field or the entire record."""
        if field_name:
            return self.schema.generation_templates.get(field_name, "")
        else:
            return self.schema.generation_templates.get('_master', "")
    
    def validate_output(self, output: Dict, field_name: str = None) -> Dict:
        """Validate generated output against schema."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        fields_to_check = [f for f in self.schema.fields if f.name == field_name] if field_name else self.schema.fields
        
        for field in fields_to_check:
            field_validation = self._validate_field(output.get(field.name), field)
            
            if not field_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(field_validation['errors'])
            
            validation_results['warnings'].extend(field_validation['warnings'])
        
        return validation_results
    
    def _validate_field(self, value: Any, field: FieldSchema) -> Dict:
        """Validate a single field value."""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check required
        if field.required and (value is None or value == ""):
            result['valid'] = False
            result['errors'].append(f"Field '{field.name}' is required")
            return result
        
        if value is None:
            return result
        
        # Type validation
        if field.type == FieldType.TEXT and not isinstance(value, str):
            result['valid'] = False
            result['errors'].append(f"Field '{field.name}' must be text")
        
        elif field.type == FieldType.NUMBER and not isinstance(value, (int, float)):
            result['valid'] = False
            result['errors'].append(f"Field '{field.name}' must be a number")
        
        elif field.type == FieldType.LIST and not isinstance(value, list):
            result['valid'] = False
            result['errors'].append(f"Field '{field.name}' must be a list")
        
        # Validation rules
        for rule, rule_value in field.validation_rules.items():
            rule_result = self._apply_validation_rule(value, rule, rule_value, field.name)
            
            if not rule_result['valid']:
                result['valid'] = False
                result['errors'].extend(rule_result['errors'])
        
        return result
    
    def _apply_validation_rule(self, value: Any, rule: str, rule_value: Any, field_name: str) -> Dict:
        """Apply a specific validation rule."""
        result = {'valid': True, 'errors': []}
        
        if rule == ValidationRule.MIN_LENGTH.value:
            # For lists, check item count; for strings, check character count
            if isinstance(value, list):
                if len(value) < rule_value:
                    result['valid'] = False
                    result['errors'].append(f"Field '{field_name}' must have at least {rule_value} items")
            else:
                if len(str(value)) < rule_value:
                    result['valid'] = False
                    result['errors'].append(f"Field '{field_name}' must be at least {rule_value} characters")
        
        elif rule == ValidationRule.MAX_LENGTH.value:
            # For lists, check item count; for strings, check character count
            if isinstance(value, list):
                if len(value) > rule_value:
                    result['valid'] = False
                    result['errors'].append(f"Field '{field_name}' must have at most {rule_value} items")
            else:
                if len(str(value)) > rule_value:
                    result['valid'] = False
                    result['errors'].append(f"Field '{field_name}' must be at most {rule_value} characters")
        
        elif rule == ValidationRule.MIN_VALUE.value:
            if isinstance(value, (int, float)) and value < rule_value:
                result['valid'] = False
                result['errors'].append(f"Field '{field_name}' must be at least {rule_value}")
        
        elif rule == ValidationRule.MAX_VALUE.value:
            if isinstance(value, (int, float)) and value > rule_value:
                result['valid'] = False
                result['errors'].append(f"Field '{field_name}' must be at most {rule_value}")
        
        elif rule == ValidationRule.PATTERN.value:
            if isinstance(value, str) and not re.match(rule_value, value):
                result['valid'] = False
                result['errors'].append(f"Field '{field_name}' does not match required pattern")
        
        elif rule == ValidationRule.CHOICES.value:
            if value not in rule_value:
                result['valid'] = False
                result['errors'].append(f"Field '{field_name}' must be one of: {', '.join(rule_value)}")
        
        return result