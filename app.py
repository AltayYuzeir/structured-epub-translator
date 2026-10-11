from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import gradio as gr

from epub_processor import (
    TranslationUnit,
    extract_epub,
    find_content_files,
    process_metadata,
    process_xhtml_file,
    rebuild_epub,
    truncate_units_by_words,
    update_epub_language,
)
from translator import TranslatorConfig, create_translator, fetch_ollama_models

CSS = """
footer {
    display: none !important;
}
"""


LANGUAGES = [
    "Arabic","Bulgarian", "Dutch", "English",
    "French", "German", 
    "Italian", "Japanese", "Korean", "Polish",
    "Portuguese", "Russian", "Spanish",
    "Turkish", 
]


def refresh_models():
    models = fetch_ollama_models()
    return gr.Dropdown(choices=models, value=models[0] if models else None)


def translate_epub(
    epub_file,
    source_language,
    target_language,
    ollama_model,
    ollama_url,
    dry_run,
    test_words,
    translate_metadata,
    max_chars,
    progress=gr.Progress(),
):
    if not epub_file:
        raise gr.Error("Please upload an EPUB file.")

    if source_language == target_language:
        raise gr.Error("Source and target languages must be different.")

    if not ollama_model:
        raise gr.Error("Please select or specify an Ollama model.")

    input_path = Path(epub_file)
    run_dir = Path(tempfile.mkdtemp(prefix="epub-translator-"))
    work_dir = run_dir / "book"
    work_dir.mkdir()

    try:
        progress(0.02, desc="Extracting EPUB...")
        extract_epub(input_path, work_dir)

        content_files = find_content_files(work_dir)
        if not content_files:
            raise gr.Error("No XHTML/HTML content was found in the EPUB.")

        config = TranslatorConfig(
            source_language=source_language,
            target_language=target_language,
            model_name=ollama_model,
            ollama_base_url=ollama_url,
            cache_dir=str(Path("cache")),
        )

        progress(0.05, desc=f"Connecting to Ollama model ({ollama_model})...")
        translator = create_translator(config)

        total_units = 0
        translated_units = 0

        for file_index, content_file in enumerate(content_files):
            def file_progress(msg: str):
                base = 0.08 + 0.82 * (file_index / max(1, len(content_files)))
                progress(min(base + 0.05, 0.95), desc=msg)

            if dry_run:
                from epub_processor import (
                    parse_xhtml,
                    build_translation_units,
                    replace_unit,
                    serialize_xhtml,
                )

                original = content_file.read_bytes()
                tree = parse_xhtml(original, content_file)
                units = build_translation_units(tree.getroot(), max_chars=int(max_chars))
                selected = truncate_units_by_words(units, int(test_words))

                for idx, unit in enumerate(selected):
                    translated = translator.translate_unit(unit)
                    if not replace_unit(unit, translated):
                        raise ValueError(
                            f"Invalid SEG placeholders in {content_file.name}, "
                            f"unit {unit.unit_id}.\n"
                            f"Expected segments: {len(unit.slots)}\n"
                            f"Model response: {translated[:1500]}"
                      )
                    progress(
                        0.08 + 0.82 * ((file_index + (idx + 1) / max(1, len(selected))) / len(content_files)),
                        desc=f"Dry run: {content_file.name} ({idx + 1}/{len(selected)})",
                    )
                    translated_units += 1
                    total_units += 1

                if selected:
                    content_file.write_bytes(serialize_xhtml(tree, original))

                break

            else:
                def translate_unit(unit: TranslationUnit) -> str:
                    return translator.translate_unit(unit)

                units, translated = process_xhtml_file(
                    content_file,
                    translate_unit,
                    progress=file_progress,
                    max_chars=int(max_chars),
                )
                total_units += units
                translated_units += translated

        if translate_metadata and not dry_run:
            progress(0.91, desc="Translating EPUB metadata...")
            meta_total, meta_translated = process_metadata(
                work_dir,
                lambda unit: translator.translate_unit(unit),
                progress=lambda msg: progress(0.93, desc=msg),
                max_chars=int(max_chars),
            )
            total_units += meta_total
            translated_units += meta_translated
            
        if not dry_run:
            progress(0.95, desc="Updating book language metadata...")
            update_epub_language(work_dir, target_language)
            
        progress(0.96, desc="Rebuilding EPUB...")

        suffix = "_test" if dry_run else f"_{target_language.lower()}"
        output_name = f"{input_path.stem}{suffix}.epub"
        # Save output directly alongside the original input EPUB file
        output_path = Path.cwd() / output_name

        rebuild_epub(work_dir, output_path)

        progress(1.0, desc="Complete")

        return str(output_path), (
            f"Completed: {translated_units} translation units. "
            f"{'Dry run mode active. ' if dry_run else ''}"
            f"Output saved as: {output_name}"
        )

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(
    title="S.P.E.L.L", css = CSS
) as demo:

    gr.Markdown(
        """
# 🗣️ Structure-Preserving EPUB Language Localizer (S.P.E.L.L)

Translate EPUB files in place using local **Ollama** models. 
Preserves images, CSS, layout, spine structure, and inner XML tags intact.
"""
    )

    with gr.Row():
        epub_file = gr.File(
            label="Source EPUB",
            file_types=[".epub"],
            type="filepath",
        )

    with gr.Row():
        with gr.Column(scale = 1):
            source_language = gr.Dropdown(
                choices=LANGUAGES,
                value="German", 
                label="Source language",
            )
        with gr.Column(scale = 1):
            target_language = gr.Dropdown(
                choices=LANGUAGES,
                value="English",
                label="Target language",
            )

    with gr.Row():
       
        with gr.Column(scale = 1):
            ollama_url = gr.Textbox(
                value="http://127.0.0.1:11434/v1",
                label="Ollama Base Endpoint"
            )
        with gr.Column(scale = 4):
            initial_models = fetch_ollama_models()
            ollama_model = gr.Dropdown(
                choices=initial_models,
                value=initial_models[0] if initial_models else "",
                label="Ollama Model",
                allow_custom_value=True
            )
        with gr.Column(scale = 1):
            refresh_btn = gr.Button("🔄 Refresh Models")

    with gr.Row():
        with gr.Column():
            dry_run = gr.Checkbox(
                value=True,
                label="Dry run (test sample first)",
            )
            
            test_words = gr.Number(
                value=1000,
                minimum=100,
                maximum=10000,
                step=100,
                label="Dry-run word limit",
            )
    
            
        with gr.Column():
            translate_metadata = gr.Checkbox(
                value=False,
                label="Translate EPUB metadata (title, description)",
            )
    
            max_chars = gr.Number(
                value=3500,
                minimum=500,
                maximum=12000,
                step=250,
                label="Maximum characters per batch group",
            )


    translate_button = gr.Button(
        "Translate EPUB",
        variant="primary",
        size="lg",
    )

    status = gr.Textbox(
        label="Status",
        interactive=False,
    )


    refresh_btn.click(
        fn=refresh_models,
        inputs=[],
        outputs=[ollama_model],
    )

    translate_button.click(
        fn=translate_epub,
        inputs=[
            epub_file,
            source_language,
            target_language,
            ollama_model,
            ollama_url,
            dry_run,
            test_words,
            translate_metadata,
            max_chars,
        ],
        outputs=[
            status,
        ],
    )


if __name__ == "__main__":
    demo.launch(
      inbrowser=True, theme=gr.themes.Monochrome(), favicon_path="language-solid-full.svg",server_port=7860)
