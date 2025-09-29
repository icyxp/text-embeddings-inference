use serde_json::Value;
use std::fmt::Write;
use std::fs;
use std::path::Path;

/// Template formatter for models that require structured prompts
pub trait TemplateFormatter {
    /// Format a query-document pair for reranking
    fn format_rerank(&self, query: &str, document: &str, instruction: Option<&str>) -> String;
}

/// Load template from tokenizer_config.json
/// Attempts to load from the given path, returns None if path doesn't exist or file can't be read
fn load_template_from_config(model_path: &str, template_key: &str) -> Option<String> {
    let config_path = Path::new(model_path).join("tokenizer_config.json");

    if let Ok(content) = fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<Value>(&content) {
            if let Some(template) = config.get(template_key) {
                return template.as_str().map(|s| s.to_string());
            }
        }
    }
    None
}

/// Generic template formatter that loads templates from tokenizer config
pub struct ConfigBasedTemplate {
    template: String,
    default_instruction: String,
}

impl ConfigBasedTemplate {
    pub fn new(template: String) -> Self {
        Self {
            template,
            default_instruction:
                "Select only the Documents that are semantically similar to the Query.".to_string(),
        }
    }

    pub fn from_model_path(model_path: &str, template_key: &str) -> Option<Self> {
        load_template_from_config(model_path, template_key).map(|template| Self::new(template))
    }
}

impl TemplateFormatter for ConfigBasedTemplate {
    fn format_rerank(&self, query: &str, document: &str, instruction: Option<&str>) -> String {
        let instruction = instruction.unwrap_or(&self.default_instruction);

        // Replace placeholders in the template
        self.template
            .replace("{instruction}", instruction)
            .replace("{query}", query)
            .replace("{document}", document)
    }
}

/// Legacy Qwen3 reranker template formatter (kept for backward compatibility)
pub struct Qwen3RerankerTemplate {
    default_instruction: String,
}

impl Qwen3RerankerTemplate {
    pub fn new() -> Self {
        Self {
            default_instruction:
                "Select only the Documents that are semantically similar to the Query.".to_string(),
        }
    }
}

impl TemplateFormatter for Qwen3RerankerTemplate {
    fn format_rerank(&self, query: &str, document: &str, instruction: Option<&str>) -> String {
        let instruction = instruction.unwrap_or(&self.default_instruction);

        let mut result = String::with_capacity(512);

        // System prompt
        result.push_str("<|im_start|>system\n");
        result.push_str("Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n");

        // User prompt with instruction, query, and document
        result.push_str("<|im_start|>user\n");
        write!(&mut result, "<Instruct>: {}\n", instruction).unwrap();
        write!(&mut result, "<Query>: {}\n", query).unwrap();
        write!(&mut result, "<Document>: {}", document).unwrap();
        result.push_str("<|im_end|>\n");

        // Assistant prompt to trigger reasoning
        result.push_str("<|im_start|>assistant\n");
        result.push_str("<think>\n\n</think>\n\n");

        result
    }
}

/// Check if a model requires template formatting by checking tokenizer config
/// This function accepts either a model path (directory containing tokenizer_config.json)
/// or a model name/id for backward compatibility
pub fn requires_template(model_path_or_id: &str) -> bool {
    // First try as a model path
    if load_template_from_config(model_path_or_id, "reranker_template").is_some()
        || load_template_from_config(model_path_or_id, "chat_template").is_some()
    {
        return true;
    }

    // Fallback to legacy model name detection for backward compatibility
    model_path_or_id.contains("Qwen3") && model_path_or_id.contains("seq-cls")
}

/// Get the appropriate template formatter for a model
/// This function accepts either a model path (directory containing tokenizer_config.json)
/// or a model name/id for backward compatibility
pub fn get_template_formatter(
    model_path_or_id: &str,
) -> Option<Box<dyn TemplateFormatter + Send + Sync>> {
    // First try to load reranker template from config (assuming it's a path)
    if let Some(formatter) =
        ConfigBasedTemplate::from_model_path(model_path_or_id, "reranker_template")
    {
        return Some(Box::new(formatter));
    }

    // Fallback to legacy detection for backward compatibility
    // This handles both cases: model_path_or_id as a path or as a model name
    let model_name = if Path::new(model_path_or_id).exists() {
        // It's a path, extract the directory name
        Path::new(model_path_or_id)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("")
    } else {
        // It's likely a model name/id
        model_path_or_id
    };

    if model_name.contains("Qwen3") && model_name.contains("seq-cls") {
        Some(Box::new(Qwen3RerankerTemplate::new()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_tokenizer_config(template_content: &str, template_key: &str) -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("tokenizer_config.json");

        let config = serde_json::json!({
            "tokenizer_class": "Qwen2Tokenizer",
            template_key: template_content
        });

        fs::write(&config_path, config.to_string()).unwrap();
        temp_dir
    }

    #[test]
    fn test_config_based_template() {
        let template_content = "<|im_start|>system\nTest system<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n";
        let temp_dir = create_test_tokenizer_config(template_content, "reranker_template");

        let formatter = ConfigBasedTemplate::from_model_path(
            temp_dir.path().to_str().unwrap(),
            "reranker_template",
        )
        .unwrap();

        let formatted = formatter.format_rerank(
            "What is Deep Learning?",
            "Deep Learning is a branch of machine learning",
            Some("Custom instruction"),
        );

        assert!(formatted.contains("Test system"));
        assert!(formatted.contains("<Instruct>: Custom instruction"));
        assert!(formatted.contains("<Query>: What is Deep Learning?"));
        assert!(formatted.contains("<Document>: Deep Learning is a branch of machine learning"));
    }

    #[test]
    fn test_requires_template_with_config() {
        let template_content = "test template with {query} and {document}";
        let temp_dir = create_test_tokenizer_config(template_content, "reranker_template");

        assert!(requires_template(temp_dir.path().to_str().unwrap()));
    }

    #[test]
    fn test_requires_template_without_config() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("tokenizer_config.json");

        let config = serde_json::json!({
            "tokenizer_class": "Qwen2Tokenizer"
        });

        fs::write(&config_path, config.to_string()).unwrap();

        assert!(!requires_template(temp_dir.path().to_str().unwrap()));
    }

    #[test]
    fn test_requires_template_legacy_model_name() {
        // Test legacy behavior with model names
        assert!(requires_template("tomaarsen/Qwen3-Reranker-0.6B-seq-cls"));
        assert!(requires_template("Qwen3-Something-seq-cls"));
        assert!(!requires_template("BAAI/bge-reranker"));
        assert!(!requires_template("Qwen3-Embed"));
    }

    #[test]
    fn test_get_template_formatter_legacy() {
        // Test legacy behavior with model names
        assert!(get_template_formatter("tomaarsen/Qwen3-Reranker-0.6B-seq-cls").is_some());
        assert!(get_template_formatter("Qwen3-Something-seq-cls").is_some());
        assert!(get_template_formatter("BAAI/bge-reranker").is_none());
        assert!(get_template_formatter("Qwen3-Embed").is_none());
    }

    #[test]
    fn test_legacy_qwen3_template() {
        let template = Qwen3RerankerTemplate::new();
        let formatted = template.format_rerank(
            "What is Deep Learning?",
            "Deep Learning is a branch of machine learning",
            None,
        );

        assert!(formatted.contains("<|im_start|>system"));
        assert!(formatted.contains("<Query>: What is Deep Learning?"));
        assert!(formatted.contains("<Document>: Deep Learning is a branch of machine learning"));
        assert!(formatted.contains("<think>"));
    }

    #[test]
    fn test_custom_instruction() {
        let template = Qwen3RerankerTemplate::new();
        let formatted =
            template.format_rerank("test query", "test doc", Some("Custom instruction"));

        assert!(formatted.contains("<Instruct>: Custom instruction"));
    }
}
