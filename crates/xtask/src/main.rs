//! This program is a development tool to be used by agents, human or otherwise.

use clap::Parser;
use scraper::{ElementRef, Html, Node, Selector};

const WGSL_SPEC_URL: &str = "https://www.w3.org/TR/WGSL/";

#[derive(clap::Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Fetch information from the WGSL specification
    WgslSpec {
        #[command(subcommand)]
        action: WgslSpecAction,
    },
}

#[derive(clap::Subcommand)]
enum WgslSpecAction {
    /// Fetch the table of contents
    Toc,
    /// Fetch a specific section by anchor ID
    Section {
        /// The section anchor ID (e.g., "vector-types",
        /// "texture-builtin-functions")
        anchor: String,
        /// Optional subsection anchor ID (e.g., "texturedimensions")
        subsection: Option<String>,
        /// Omit subsections, only fetch the immediate section content
        #[arg(long)]
        shallow: bool,
    },
}

fn main() {
    env_logger::builder().init();
    let cli = Cli::parse();

    match cli.command {
        Commands::WgslSpec { action } => match action {
            WgslSpecAction::Toc => {
                if let Err(e) = fetch_toc() {
                    eprintln!("Error fetching TOC: {e}");
                    std::process::exit(1);
                }
            }
            WgslSpecAction::Section {
                anchor,
                subsection,
                shallow,
            } => {
                if let Err(e) = fetch_section(&anchor, subsection.as_deref(), shallow) {
                    eprintln!("Error fetching section: {e}");
                    std::process::exit(1);
                }
            }
        },
    }
}

fn fetch_toc() -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::blocking::get(WGSL_SPEC_URL)?;
    let html = response.text()?;
    let document = Html::parse_document(&html);

    let nav_selector = Selector::parse("nav#toc").unwrap();
    let li_selector = Selector::parse("li").unwrap();
    let a_selector = Selector::parse("a").unwrap();
    let secno_selector = Selector::parse(".secno").unwrap();
    let content_selector = Selector::parse(".content").unwrap();

    let nav = document
        .select(&nav_selector)
        .next()
        .ok_or("Could not find TOC nav element")?;

    for li in nav.select(&li_selector) {
        if let Some(a) = li.select(&a_selector).next() {
            let href = a.value().attr("href").unwrap_or("");
            let secno = a
                .select(&secno_selector)
                .next()
                .map(|el| el.text().collect::<String>())
                .unwrap_or_default();
            let content = a
                .select(&content_selector)
                .next()
                .map(|el| el.text().collect::<String>())
                .unwrap_or_default();

            // Calculate indent based on section number depth
            let depth = secno.matches('.').count();
            let indent = "  ".repeat(depth);

            println!("{indent}{secno} {content} ({href})");
        }
    }

    Ok(())
}

fn fetch_section(
    anchor: &str,
    subsection: Option<&str>,
    shallow: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::blocking::get(WGSL_SPEC_URL)?;
    let html = response.text()?;
    let document = Html::parse_document(&html);

    // Determine which anchor to look for
    let target_anchor = subsection.unwrap_or(anchor);

    // Find the heading element with the target anchor ID
    let heading_selector = Selector::parse(&format!(
        "h1#{target_anchor}, h2#{target_anchor}, h3#{target_anchor}, h4#{target_anchor}, \
         h5#{target_anchor}, h6#{target_anchor}"
    ))
    .map_err(|e| format!("Invalid selector: {e:?}"))?;

    let heading = document.select(&heading_selector).next().ok_or_else(|| {
        format!(
            "Could not find section with anchor '#{target_anchor}'.\nRun 'cargo xtask wgsl-spec \
             toc' to see available sections."
        )
    })?;

    // Determine the heading level (h2 -> 2, h3 -> 3, etc.)
    let heading_level = get_heading_level(heading).ok_or("Could not determine heading level")?;

    // Determine the boundary level - if shallow, stop at same level or higher (any
    // subsection) If not shallow, stop only at same level or higher
    let boundary_level = if shallow {
        heading_level + 1 // Stop at any subsection heading
    } else {
        heading_level // Stop only at same or higher level
    };

    // Build HTML content by collecting sibling elements
    let mut content_html = String::new();

    // Include the heading itself
    content_html.push_str(&heading.html());

    // Walk through siblings after the heading
    let mut current = heading.next_sibling();
    while let Some(node) = current {
        match node.value() {
            Node::Element(el) => {
                let tag_name = el.name();
                // Check if this is a heading that marks the end of our section
                if let Some(level) = heading_level_from_tag(tag_name) {
                    if shallow {
                        // In shallow mode, stop at any heading (subsection or higher)
                        if level <= boundary_level {
                            break;
                        }
                    } else {
                        // In deep mode, stop only at same level or higher
                        if level <= heading_level {
                            break;
                        }
                    }
                }
                // Include this element
                if let Some(element_ref) = ElementRef::wrap(node) {
                    content_html.push_str(&element_ref.html());
                }
            }
            Node::Text(text) => {
                content_html.push_str(text);
            }
            _ => {}
        }
        current = node.next_sibling();
    }

    // Convert HTML to text/markdown using html2text
    let text = html2text::from_read(content_html.as_bytes(), 100)?;
    println!("{text}");

    Ok(())
}

/// Get the heading level from an ElementRef (e.g., h2 -> 2, h3 -> 3)
fn get_heading_level(element: ElementRef) -> Option<u8> {
    heading_level_from_tag(element.value().name())
}

/// Get the heading level from a tag name (e.g., "h2" -> 2, "h3" -> 3)
fn heading_level_from_tag(tag_name: &str) -> Option<u8> {
    match tag_name {
        "h1" => Some(1),
        "h2" => Some(2),
        "h3" => Some(3),
        "h4" => Some(4),
        "h5" => Some(5),
        "h6" => Some(6),
        _ => None,
    }
}
