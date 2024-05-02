import axios from "axios";

const BASE_URL = "https://api.openai.com/v1";

export const fetchChatGPTResponse = async (prompt) => {
  try {
    console.log(process.env.REACT_APP_GPT_KEY);
    const response = await axios.post(
      `${BASE_URL}/chat/completions`,
      {
        model: "gpt-4", // or any specific model you are using
        messages: [{ role: "user", content: prompt }],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.REACT_APP_GPT_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error calling OpenAI API:", error);
    throw error;
  }
};
