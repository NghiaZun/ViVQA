def call_teacher_qwen(image_path: str, question: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot open image {image_path}: {e}")
        return {"answer": "", "reasoning": "", "raw": ""}

    user_prompt = build_fewshot_prompt(question)
    
    # ✅ CRITICAL FIX: Use Qwen2-VL's conversation format with image placeholder
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # Pass PIL image directly
                },
                {
                    "type": "text",
                    "text": f"Hãy quan sát ảnh và trả lời câu hỏi theo đúng định dạng:\n{user_prompt}\n\nBắt đầu bằng:\nAnswer: ...\nReasoning: ..."
                }
            ]
        }
    ]

    try:
        # ✅ Apply chat template - this creates proper image tokens
        text_prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # ✅ Process with the templated prompt
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # ✅ Generate with appropriate parameters
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50
            )

        # ✅ Decode only the generated part (skip input tokens)
        generated_ids = output[:, inputs.input_ids.shape[1]:]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # DEBUG
        print("=" * 80)
        print(f"[DEBUG] Question: {question}")
        print(f"[DEBUG] Raw output:\n{text}\n")

        # Parse answer & reasoning
        answer, reasoning = "", ""
        for line in text.splitlines():
            line_clean = line.strip()
            if line_clean.lower().startswith("answer:"):
                answer = line_clean.split(":", 1)[1].strip()
            elif line_clean.lower().startswith("reasoning:"):
                reasoning = line_clean.split(":", 1)[1].strip()

        # Fallback parsing
        if not answer and text:
            parts = text.split("\n", 1)
            answer = parts[0].strip()
            if len(parts) > 1:
                reasoning = parts[1].strip()

        return {"answer": answer, "reasoning": reasoning, "raw": text}

    except Exception as e:
        print(f"[WARN] Generation failed for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"answer": "", "reasoning": "", "raw": ""}
