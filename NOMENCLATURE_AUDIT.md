# PCC Data Pipeline - Nomenclature Coherence Audit

## 🎯 **System Identity Standardization**

### **Current Inconsistencies**
| Location | Current Name | Issue |
|----------|--------------|-------|
| `src/data_pipeline.py` | "Enterprise Data Pipeline" | Generic, legacy |
| `README.md` | "Privacy Case Classifier (PCC) - Data Pipeline" | Correct but verbose |
| `config.yaml` | "Enhanced Enterprise Data Pipeline" | Outdated |
| `docker-compose.yml` | "Enterprise Data Pipeline Stack" | Generic |
| Prometheus metrics | `pipeline_name='pcc_data_pipeline'` | Inconsistent format |

### **✅ Proposed Standard**
- **Official Name**: `PCC Data Pipeline`
- **Full Name**: `Privacy Case Classifier (PCC) Data Pipeline`
- **Code Identifier**: `pcc_data_pipeline`
- **Container Name**: `pcc-data-pipeline`
- **Class Name**: `PCCDataPipeline` (rename from `EnterpriseDataPipeline`)

## 🔧 **Process Terminology Standardization**

### **Current Inconsistencies**

#### **Feature Processing**
| Context | Current Term | Issue |
|---------|--------------|-------|
| Code methods | `extract_features()` | Implies extraction only |
| Documentation | "Feature Engineering" | Implies creation/transformation |
| Stage names | "feature_extraction" | Inconsistent with docs |
| Config keys | `extract_text_features` | Verb form, unclear |

**✅ Proposed Standard**: Use **"Feature Engineering"** consistently
- **Method**: `engineer_features()` or `extract_and_engineer_features()`
- **Stage Name**: `feature_engineering`
- **Config**: `feature_engineering: {enabled: true, ...}`
- **Documentation**: Always "Feature Engineering"

#### **Embedding Processing**
| Context | Current Term | Status |
|---------|--------------|--------|
| Code methods | `generate_embeddings()` | ✅ Consistent |
| Stage names | "embedding_generation" | ✅ Consistent |
| Config keys | `generate_embeddings` | ✅ Consistent |
| Documentation | "Embedding Generation" | ✅ Consistent |

**Status**: ✅ Already coherent

### **Stage Name Standardization**
| Process | Current Code | Current Docs | Proposed Standard |
|---------|--------------|--------------|-------------------|
| Data creation | "synthetic_generation" | "Enhanced Data Generation" | `synthetic_data_generation` |
| Feature work | "feature_extraction" | "Feature Engineering" | `feature_engineering` |
| Embedding work | "embedding_generation" | "Embedding Generation" | `embedding_generation` ✅ |
| Quality checks | "validation" | "Enterprise Validation" | `data_validation` |
| Data sampling | "sampling" | "Intelligent Sampling" | `data_sampling` |

## 📝 **Configuration Key Standardization**

### **Current Inconsistencies**
```yaml
# Mixed naming patterns:
extract_text_features: true        # verb_adjective_noun
generate_embeddings: true          # verb_noun
feature_extractor: {...}          # noun_noun
validation_enabled: true          # noun_adjective
lineage: {enabled: true}          # noun + nested enabled
contracts: {enabled: true}        # noun + nested enabled
```

### **✅ Proposed Standard Pattern**
```yaml
# Consistent {process}: {enabled: bool, config: {...}} pattern
feature_engineering:
  enabled: true
  config: {...}

embedding_generation:
  enabled: true  
  config: {...}

data_validation:
  enabled: true
  config: {...}

synthetic_data:
  enabled: true
  config: {...}

lineage_tracking:
  enabled: true
  config: {...}

pipeline_contracts:
  enabled: true
  config: {...}
```

## 🏗️ **Class and Method Naming**

### **Current Inconsistencies**
| Current | Issue | Proposed |
|---------|-------|----------|
| `EnterpriseDataPipeline` | Generic | `PCCDataPipeline` |
| `extract_features()` | Ambiguous | `engineer_features()` |
| `PCCEcosystemContracts` | ✅ Good | Keep |
| `TextFeatureExtractor` | Inconsistent with "engineering" | `TextFeatureEngineer` |

### **Method Name Patterns**
| Process | Current | Proposed Standard |
|---------|---------|-------------------|
| Feature work | `extract_features()` | `engineer_features()` |
| Embedding work | `generate_embeddings()` | `generate_embeddings()` ✅ |
| Data creation | `generate_synthetic_data()` | `generate_synthetic_data()` ✅ |
| Validation | `validate_data()` | `validate_data()` ✅ |

## 📊 **Metrics and Monitoring Names**

### **Current Issues**
```python
# Inconsistent metric naming:
pipeline_name='pcc_data_pipeline'     # snake_case
check_type='completeness'            # lowercase
feature_category=category            # variable case
```

### **✅ Proposed Standard**
```python
# Consistent metric labeling:
pipeline_name='pcc-data-pipeline'    # kebab-case for services
component='feature-engineering'       # kebab-case components  
check_type='completeness'            # lowercase types ✅
dataset_type='training-data'         # kebab-case data types
```

## 🎯 **Priority Fix List**

### **High Priority (Breaking Changes)**
1. Rename `EnterpriseDataPipeline` → `PCCDataPipeline`
2. Rename `extract_features()` → `engineer_features()`
3. Update all "Enterprise Data Pipeline" → "PCC Data Pipeline"
4. Standardize stage names across code/docs
5. Rename `TextFeatureExtractor` → `TextFeatureEngineer`

### **Medium Priority (Configuration)**
1. Restructure config.yaml with consistent patterns
2. Update all config keys to match new standards
3. Update environment variables and Docker configs

### **Low Priority (Documentation)**
1. Ensure consistent terminology in all docs
2. Update metric names for consistency
3. Align CLI argument names with new standards

## 🚦 **Implementation Strategy**

### **Phase 1: Core Renaming**
- Update class names and primary identifiers
- Update configuration structure
- Update README and main documentation

### **Phase 2: Method Alignment** 
- Rename methods to match terminology
- Update stage names in lineage tracking
- Update metric names

### **Phase 3: Documentation Polish**
- Ensure all docs use consistent terminology
- Update CLI help text and error messages
- Final coherence review

## ✅ **Validation Checklist**

- [ ] System has single, clear identity
- [ ] Feature processing uses consistent terminology
- [ ] Stage names match between code and docs
- [ ] Configuration follows consistent patterns
- [ ] Class names reflect actual purpose
- [ ] Method names align with documentation
- [ ] Metrics use consistent naming
- [ ] CLI arguments match internal naming
- [ ] Error messages use standard terminology
- [ ] All documentation uses agreed terms

---

**Goal**: Transform from "enterprise-data-pipeline with identity crisis" to "PCC Data Pipeline with crystal-clear, professional nomenclature" 